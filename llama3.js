require("dotenv").config();
const { Telegraf, Markup } = require("telegraf");
const { createClient } = require("@supabase/supabase-js");

const { Ollama } = require("@langchain/ollama");
const { OllamaEmbeddings } = require("@langchain/ollama");
const {
  SupabaseVectorStore,
} = require("@langchain/community/vectorstores/supabase");

// --- 0. VALIDASI ENVIRONMENT VARIABLE ---
const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_PRIVATE_KEY;

if (!BOT_TOKEN || !SUPABASE_URL || !SUPABASE_KEY) {
  console.error(
    "❌ GAGAL START: Environment variable belum lengkap. Cek file .env (TELEGRAM_BOT_TOKEN, SUPABASE_URL, SUPABASE_PRIVATE_KEY).",
  );
  process.exit(1);
}

// --- 1. KONFIGURASI ---
// handlerTimeout diturunkan ke nilai wajar (dulu 9.000.000 ms = 2.5 jam)
const bot = new Telegraf(BOT_TOKEN, { handlerTimeout: 360_000 });

const supabaseClient = createClient(SUPABASE_URL, SUPABASE_KEY);

const NAMA_MODEL = "llama3.2:latest";

const llm = new Ollama({
  baseUrl: "http://localhost:11434",
  model: NAMA_MODEL,
  temperature: 0,
  requestOptions: {
    num_thread: 4,
    num_predict: 600,
    num_ctx: 4096, // dinaikkan dari 2028 (typo) agar tidak overflow
    repeat_penalty: 1.1,
  },
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
});

let vectorStore;

// --- 2. STATE MANAGEMENT PER USER (IN-MEMORY) ---
// Setiap user punya: riwayat chat (untuk konteks LLM) DAN status (untuk logika bisnis).
// Status inilah yang menentukan kapan insert ke DB terjadi -- BUKAN keputusan LLM.
// Status: "idle" | "mengisi_form" | "menunggu_konfirmasi"
const sesiUser = {};

function ambilSesi(userId) {
  if (!sesiUser[userId]) {
    sesiUser[userId] = {
      status: "idle",
      riwayat: [],
      dataForm: null, // menyimpan data yang sedang dikonfirmasi
      terakhirAktif: Date.now(),
    };
  }
  sesiUser[userId].terakhirAktif = Date.now();
  return sesiUser[userId];
}

// Bersihkan sesi yang tidak aktif > 30 menit, supaya object tidak menumpuk terus (memory leak)
setInterval(
  () => {
    const batas = 30 * 60 * 1000;
    const sekarang = Date.now();
    for (const userId in sesiUser) {
      if (sekarang - sesiUser[userId].terakhirAktif > batas) {
        delete sesiUser[userId];
      }
    }
  },
  10 * 60 * 1000,
); // cek tiap 10 menit

// --- 3. FUNGSI KONEKSI KE SUPABASE ---
async function siapkanLemariSupabase() {
  console.log("🗄️ [1/1] Menghubungkan bot ke Knowledge Base di Supabase...");
  vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: "knowledge_base",
    queryName: "match_knowledge_base",
  });
  console.log("✅ SISTEM SIAP: Berhasil terhubung ke database pengetahuan.\n");
}

// --- 4. HELPER: HAPUS PESAN LOADING DENGAN AMAN ---
async function hapusPesanAman(ctx, messageId) {
  if (!messageId) return;
  try {
    await ctx.telegram.deleteMessage(ctx.chat.id, messageId);
  } catch (e) {
    // diabaikan, tidak fatal
  }
}

// --- 5. HELPER: PANGGIL LLM DENGAN TIMEOUT ---
function invokeDenganTimeout(prompt, timeoutMs = 90000) {
  return Promise.race([
    llm.invoke(prompt),
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error("Timeout: model tidak merespons")),
        timeoutMs,
      ),
    ),
  ]);
}

const PESAN_TIDAK_TERSEDIA =
  "Mohon maaf, informasi tersebut tidak tersedia di data resmi kami.";
const PESAN_SIBUK =
  "Mohon maaf, sistem sedang sibuk. Silakan coba beberapa saat lagi.";

// --- 6. PROMPT ROUTER (TAHAP 1: KLASIFIKASI NIAT) ---
// Prompt pendek dan fokus, hanya untuk menentukan niat -- bukan mengerjakan tugasnya.
function buatPromptRouter(pesanUser, statusSaatIni) {
  return `Anda adalah pengklasifikasi niat pesan untuk chatbot sekolah.
Balas HANYA dengan satu kata dari daftar berikut, tanpa penjelasan tambahan:
- INFO (jika user bertanya seputar sekolah: syarat, biaya, ekskul, lokasi, dll)
- DAFTAR (jika user menyatakan ingin mendaftar/masuk sekolah)
- ISI_FORM (jika status saat ini "mengisi_form" dan pesan berisi data diri seperti nama/NIK/alamat)
- KONFIRMASI_YA (jika status saat ini "menunggu_konfirmasi" dan user menyetujui, misal "ya", "benar", "betul")
- KONFIRMASI_TIDAK (jika status saat ini "menunggu_konfirmasi" dan user menolak/ingin mengubah data)
- LAINNYA (jika tidak masuk kategori manapun)

STATUS SAAT INI: ${statusSaatIni}
PESAN USER: "${pesanUser}"

KATEGORI:`;
}

// --- 7. PROMPT INFO (RAG BIASA) ---
const SYSTEM_PROMPT_INFO = `Anda adalah Admin Virtual SMP Elim Kairos yang ramah dan informatif.

ATURAN JAWABAN:
1. Jawab HANYA berdasarkan teks di bagian DATA. Jangan gunakan pengetahuan umum di luar DATA.
2. Gunakan format poin (-) dengan kalimat pendek dan jelas, beri jarak antar baris.
3. Jangan bahas topik lain selain yang ditanyakan.
4. Jika DATA tidak memuat jawaban, balas persis: "${PESAN_TIDAK_TERSEDIA}"
5. DILARANG keras menambah, mengarang, atau berasumsi di luar DATA.

CONTOH:
PERTANYAAN: Berapa biaya pendaftaran?
DATA: Biaya pendaftaran Rp500.000, SPP bulanan Rp350.000.
JAWABAN:
- Biaya pendaftaran: Rp500.000
- SPP bulanan: Rp350.000`;

function buatPromptInfo(teksContekan, userQuestion) {
  if (!teksContekan || !teksContekan.trim()) return null;
  return `${SYSTEM_PROMPT_INFO}

DATA:
${teksContekan}

PERTANYAAN:
${userQuestion}

JAWABAN:`;
}

// --- 8. PROMPT EKSTRAKSI DATA FORM (FORMAT JSON, BUKAN PIPE-DELIMITED) ---
const TEMPLATE_FORM = `Halo! Selamat datang di layanan pendaftaran daring SMP Elim Kairos. Silakan copy-paste dan lengkapi formulir pendaftaran berikut:
1. Nama Lengkap :
2. Jenis Kelamin (L/P) :
3. NIK :
4. Tempat & Tanggal Lahir :
5. Agama :
6. Alamat Lengkap (Jalan, RT/RW, Desa, Kec) :
7. Transportasi ke Sekolah (Jalan Kaki/Motor/Lainnya) :
8. No. Telp / HP / WA :`;

function buatPromptEkstraksi(pesanUser) {
  return `Ekstrak data pendaftaran dari pesan user berikut ke dalam format JSON.
Jika ada field yang tidak diisi/tidak ditemukan, isi dengan string kosong "".
Balas HANYA dengan JSON, tanpa teks lain, tanpa markdown code block.

Format JSON yang WAJIB diikuti persis:
{"nama_lengkap":"","jenis_kelamin":"","nik":"","ttl":"","agama":"","alamat":"","transportasi":"","no_hp":""}

PESAN USER:
"${pesanUser}"

JSON:`;
}

function formatKonfirmasi(data) {
  return `Berikut data Anda:
Nama: ${data.nama_lengkap}
Jenis Kelamin: ${data.jenis_kelamin}
NIK: ${data.nik}
TTL: ${data.ttl}
Agama: ${data.agama}
Alamat: ${data.alamat}
Transportasi: ${data.transportasi}
No HP: ${data.no_hp}

Apakah data di atas sudah benar? (Ketik YA jika benar, atau TIDAK jika ingin mengisi ulang)`;
}

// --- 9. FUNGSI UTAMA: JALUR INFO (RAG) ---
async function tanganiInfo(ctx, userQuestion, loadingMsg) {
  try {
    const hasilPencarian = await vectorStore.similaritySearch(userQuestion, 3);
    const teksContekan = hasilPencarian.map((d) => d.pageContent).join("\n\n");
    const promptFinal = buatPromptInfo(teksContekan, userQuestion);

    if (!promptFinal) {
      await hapusPesanAman(ctx, loadingMsg?.message_id);
      await ctx.reply(PESAN_TIDAK_TERSEDIA);
      return;
    }

    const aiResponse = await invokeDenganTimeout(promptFinal);
    const cleanResponse = aiResponse.replace(/\*\*/g, "").trim();

    await hapusPesanAman(ctx, loadingMsg?.message_id);
    await ctx.reply(cleanResponse);
  } catch (error) {
    console.error("Error pada jalur INFO:", error.message);
    await hapusPesanAman(ctx, loadingMsg?.message_id);
    await ctx.reply(PESAN_SIBUK);
  }
}

// --- 10. FUNGSI UTAMA: JALUR AGENT PENDAFTARAN (STATE MACHINE) ---
async function tanganiPendaftaran(
  ctx,
  userId,
  pesanAsli,
  kategori,
  sesi,
  loadingMsg,
) {
  try {
    if (kategori === "DAFTAR") {
      sesi.status = "mengisi_form";
      await hapusPesanAman(ctx, loadingMsg?.message_id);
      await ctx.reply(TEMPLATE_FORM);
      return;
    }

    if (kategori === "ISI_FORM") {
      const promptEkstraksi = buatPromptEkstraksi(pesanAsli);
      const rawJson = await invokeDenganTimeout(promptEkstraksi);

      let data;
      try {
        // Bersihkan kemungkinan markdown fence yang kadang ditambahkan LLM
        const bersih = rawJson.replace(/```json|```/g, "").trim();
        data = JSON.parse(bersih);
      } catch (parseErr) {
        await hapusPesanAman(ctx, loadingMsg?.message_id);
        await ctx.reply(
          "⚠️ Maaf, format data belum bisa terbaca. Mohon isi ulang formulir sesuai contoh di atas.",
        );
        return;
      }

      sesi.dataForm = data;
      sesi.status = "menunggu_konfirmasi";

      await hapusPesanAman(ctx, loadingMsg?.message_id);
      await ctx.reply(formatKonfirmasi(data));
      return;
    }

    if (kategori === "KONFIRMASI_YA" && sesi.status === "menunggu_konfirmasi") {
      const data = sesi.dataForm;

      // Catatan demo: validasi format NIK/HP sengaja belum diterapkan (tahap demo).
      const { error } = await supabaseClient.from("calon_siswa").insert([
        {
          nama_lengkap: data.nama_lengkap,
          jenis_kelamin: data.jenis_kelamin,
          nik: data.nik,
          ttl: data.ttl,
          agama: data.agama,
          alamat: data.alamat,
          transportasi: data.transportasi,
          no_hp: data.no_hp,
        },
      ]);

      if (error) throw new Error(`DB Error: ${error.message}`);

      const namaTampil = data.nama_lengkap || "Anda";
      sesi.status = "idle";
      sesi.dataForm = null;
      sesi.riwayat = [];

      await hapusPesanAman(ctx, loadingMsg?.message_id);
      await ctx.reply(
        `✅ Pendaftaran Awal Berhasil!\n\nData atas nama ${namaTampil} telah masuk ke sistem kami. Silakan datang ke tata usaha sekolah membawa dokumen lampiran (Fotokopi KK, Akta Kelahiran, dll) untuk menyelesaikan pendaftaran Anda. Terima kasih!`,
      );
      return;
    }

    if (
      kategori === "KONFIRMASI_TIDAK" &&
      sesi.status === "menunggu_konfirmasi"
    ) {
      sesi.status = "mengisi_form";
      sesi.dataForm = null;
      await hapusPesanAman(ctx, loadingMsg?.message_id);
      await ctx.reply(
        "Baik, silakan kirim ulang data formulir yang benar:\n\n" +
          TEMPLATE_FORM,
      );
      return;
    }

    // Fallback: kategori tidak cocok dengan status saat ini
    await hapusPesanAman(ctx, loadingMsg?.message_id);
    await ctx.reply(
      "Maaf, saya belum menangkap maksud Anda. Ketik ulang niat pendaftaran Anda, atau data formulir sesuai format yang diminta.",
    );
  } catch (error) {
    console.error("Error pada jalur PENDAFTARAN:", error.message);
    await hapusPesanAman(ctx, loadingMsg?.message_id);
    await ctx.reply(PESAN_SIBUK);
  }
}

// --- 11. PROSES PESAN (MENU UTAMA) ---
bot.start(async (ctx) => {
  const pesanSambut =
    "Halo! Saya Asisten Virtual SMP Elim Kairos. Apa yang ingin Anda ketahui?";
  const tombol = Markup.inlineKeyboard([
    [
      Markup.button.callback("📝 Syarat Daftar", "tanya_syarat"),
      Markup.button.callback("💰 Info Biaya", "tanya_biaya"),
    ],
    [
      Markup.button.callback("🏆 Ekstrakurikuler", "tanya_ekskul"),
      Markup.button.callback("📍 Lokasi Sekolah", "tanya_lokasi"),
    ],
  ]);
  await ctx.reply(pesanSambut, tombol);
});

// --- 12. PENANGANAN KLIK TOMBOL (ACTION) ---
const PERTANYAAN_TOMBOL = {
  syarat: "Apa saja syarat pendaftaran siswa baru?",
  biaya: "Berapa rincian biaya pendaftaran dan SPP?",
  ekskul: "Sebutkan ekstrakurikuler unggulan yang ada.",
  lokasi: "Di mana alamat atau lokasi sekolah?",
};

bot.action(/tanya_(.+)/, async (ctx) => {
  await ctx.answerCbQuery();
  const aksi = ctx.match[1];
  const userQuestion = PERTANYAAN_TOMBOL[aksi];

  if (!userQuestion) return ctx.reply(PESAN_TIDAK_TERSEDIA);

  const loadingMsg = await ctx.reply(`⏳ Mencari info tentang ${aksi}...`);
  await ctx.sendChatAction("typing");
  await tanganiInfo(ctx, userQuestion, loadingMsg);
});

// --- 13. PENANGANAN KETIKAN TEKS MANUAL (ROUTER + AGENT) ---
bot.on("text", async (ctx) => {
  const userId = ctx.from.id;
  const pesanAsli = ctx.message.text;
  const userQuestion = pesanAsli.toLowerCase().trim();

  if (userQuestion === "/start") return;

  const sesi = ambilSesi(userId);

  // 🚀 JALUR CEPAT SAPAAN (hanya berlaku kalau user sedang idle, bukan di tengah isi form)
  const jumlahKata = userQuestion.split(/\s+/).length;
  const tangkapSapaan = userQuestion.match(
    /\b(pagi|siang|sore|malam|assalamualaikum|hai|halo|hallo|ping|p)\b/i,
  );
  const adaTanya =
    /\b(apa|apakah|berapa|mana|dimana|kapan|gimana|bagaimana|syarat|biaya|ekskul|lokasi|daftar|spp|masuk)\b/i.test(
      userQuestion,
    );

  if (sesi.status === "idle" && jumlahKata <= 6 && tangkapSapaan && !adaTanya) {
    let sapaanBalasan = "Halo!";
    const kataSapa = tangkapSapaan[1].toLowerCase();
    if (["pagi", "siang", "sore", "malam"].includes(kataSapa)) {
      sapaanBalasan = `Selamat ${kataSapa.charAt(0).toUpperCase() + kataSapa.slice(1)}!`;
    } else if (kataSapa === "assalamualaikum") {
      sapaanBalasan = "Waalaikumsalam warahmatullah!";
    }
    return ctx.reply(
      `${sapaanBalasan} 👋 Saya Asisten Virtual SMP Elim Kairos. Silakan ketik pertanyaan Anda atau niat untuk mendaftar.`,
    );
  }

  // 🐢 JALUR LAMBAT: tentukan dulu kategori niat via router, baru eksekusi
  const loadingMsg = await ctx.reply("⏳ Memproses pesan Anda...");
  await ctx.sendChatAction("typing");

  const startTime = new Date();
  console.log(
    `[${startTime.toLocaleTimeString()}] Pesan dari ${userId} (status: ${sesi.status}): "${pesanAsli}"`,
  );

  try {
    let kategori;

    // Kalau status sudah spesifik (mengisi_form / menunggu_konfirmasi), tidak perlu
    // klasifikasi ulang untuk kasus paling umum -- langsung asumsikan sesuai status,
    // kecuali polanya cocok dengan kata kunci info/daftar/konfirmasi.
    if (
      sesi.status === "menunggu_konfirmasi" &&
      /^(ya|benar|betul|iya)\b/i.test(userQuestion)
    ) {
      kategori = "KONFIRMASI_YA";
    } else if (
      sesi.status === "menunggu_konfirmasi" &&
      /^(tidak|salah|belum)\b/i.test(userQuestion)
    ) {
      kategori = "KONFIRMASI_TIDAK";
    } else {
      const promptRouter = buatPromptRouter(pesanAsli, sesi.status);
      const rawKategori = await invokeDenganTimeout(promptRouter, 60000);
      kategori = rawKategori.replace(/[^A-Z_]/g, "").trim();
    }

    console.log(`   -> Kategori terdeteksi: ${kategori}`);

    if (kategori === "INFO") {
      await tanganiInfo(ctx, pesanAsli, loadingMsg);
    } else if (
      ["DAFTAR", "ISI_FORM", "KONFIRMASI_YA", "KONFIRMASI_TIDAK"].includes(
        kategori,
      )
    ) {
      await tanganiPendaftaran(
        ctx,
        userId,
        pesanAsli,
        kategori,
        sesi,
        loadingMsg,
      );
    } else {
      // LAINNYA atau kategori tidak terbaca dengan baik
      await hapusPesanAman(ctx, loadingMsg?.message_id);
      if (sesi.status === "mengisi_form") {
        // fallback aman: kalau lagi di tengah isi form, coba tetap ekstrak
        await tanganiPendaftaran(
          ctx,
          userId,
          pesanAsli,
          "ISI_FORM",
          sesi,
          null,
        );
      } else {
        await ctx.reply(
          "Maaf, saya belum memahami maksud Anda. Anda bisa bertanya seputar sekolah atau ketik 'daftar' untuk mulai pendaftaran.",
        );
      }
    }

    const endTime = new Date();
    console.log(
      `[${endTime.toLocaleTimeString()}] Selesai dalam ${endTime - startTime}ms\n`,
    );
  } catch (error) {
    console.error("Error pada jalur utama:", error.message);
    await hapusPesanAman(ctx, loadingMsg?.message_id);
    await ctx.reply(PESAN_SIBUK);
  }
});

// --- 14. RUN ---
async function main() {
  try {
    await siapkanLemariSupabase();
    await bot.launch();
    console.log("===========================================");
    console.log(` ✅ SISTEM AKTIF (${NAMA_MODEL}, AGENT & RAG)`);
    console.log(" 🤖 Bot siap melayani pendaftaran SMP      ");
    console.log("===========================================");
  } catch (error) {
    console.error("❌ GAGAL START:", error.message);
    process.exit(1);
  }
}

main();

// --- 15. GRACEFUL SHUTDOWN ---
process.once("SIGINT", () => bot.stop("SIGINT"));
process.once("SIGTERM", () => bot.stop("SIGTERM"));
