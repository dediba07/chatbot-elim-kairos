require("dotenv").config();
const fs = require("fs");
const { createClient } = require("@supabase/supabase-js");

// Menggunakan jalur impor versi terbaru (Sama seperti di llama3.js)
const { RecursiveCharacterTextSplitter } = require("@langchain/textsplitters");
const { OllamaEmbeddings } = require("@langchain/ollama");
const {
  SupabaseVectorStore,
} = require("@langchain/community/vectorstores/supabase");

async function jalankanIngestion() {
  try {
    console.log("1. Membaca file knowledge_base.md...");
    const text = fs.readFileSync("knowledge_base.md", "utf8");

    console.log("2. Memecah teks (Chunking)...");
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 150,
    });
    const docs = await splitter.createDocuments([text]);
    console.log(
      `--> Berhasil dipecah menjadi ${docs.length} potongan dokumen.`,
    );

    console.log("3. Menghubungkan ke Supabase...");
    const client = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_PRIVATE_KEY,
    );

    console.log(
      "4. Proses Embedding via Ollama dan Upload ke Supabase (Harap tunggu)...",
    );
    const embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text", // Sudah disamakan dengan llama3.js
      baseUrl: "http://localhost:11434",
    });

    // Mengunggah ke tabel Supabase
    await SupabaseVectorStore.fromDocuments(docs, embeddings, {
      client,
      tableName: "knowledge_base",
      queryName: "match_knowledge_base",
    });

    console.log(
      "✅ SELESAI! Seluruh data basis pengetahuan berhasil diunggah ke database.",
    );
  } catch (error) {
    console.error("❌ Terjadi kesalahan:", error);
  }
}

jalankanIngestion();
