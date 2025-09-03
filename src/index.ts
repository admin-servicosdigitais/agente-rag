import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { LocalVectorStore } from "./services/store";
import { ensureSourcesDir, updateIndex } from "./services/rag";
import { embedText } from "./services/bedrock";
import { answerWithRAG } from "./services/agent";

async function main() {
  console.log("\n[RAG CLI • Base local • /atualizar para indexar .md na pasta ./fontes]");
  await ensureSourcesDir();

  const store = new LocalVectorStore();
  const rl = readline.createInterface({ input, output });

  console.log("Comandos:");
  console.log("  /atualizar  -> indexa/atualiza embeddings dos .md em ./fontes");
  console.log("  /sair       -> encerra\n");

  while (true) {
    const q = (await rl.question("Pergunte ou comando: ")).trim();
    if (!q) continue;

    if (q === "/sair") break;

    if (q === "/atualizar") {
      try {
        console.log("Indexando .md da pasta ./fontes ...");
        const res = await updateIndex(store);
        console.log(
          `Atualização concluída. Processados: ${res.processed}, Ignorados: ${res.skipped}, Novos chunks: ${res.totalChunks}`
        );
        console.log(`Estado atual -> arquivos: ${res.stats.files}, chunks: ${res.stats.chunks}\n`);
      } catch (err: any) {
        console.error("Falha ao atualizar índice:", err?.message || err);
        if (err?.name === "AccessDeniedException") {
          console.log("\nVerifique permissões e habilitação do modelo de embeddings (Titan) na região configurada.");
        }
      }
      continue;
    }

    // Consulta normal (RAG)
    try {
      const qVec = await embedText(q);
      const { answer, used } = await answerWithRAG(store, qVec, q);
      console.log("\nResposta:\n", answer, "\n");

      if (used.length) {
        const linhas = used.map((u, i) => `  [${i + 1}] ${u.sourcePath} (chunk ${u.chunkIndex})`).join("\n");
        console.log("Fontes utilizadas:\n" + linhas + "\n");
      } else {
        console.log("Nenhum trecho recuperado do índice. Tente /atualizar.\n");
      }
    } catch (err: any) {
      console.error("Erro ao responder:", err?.message || err);
      if (err?.name === "AccessDeniedException") {
        console.log("\nVerifique:");
        console.log("- Modelo de embeddings habilitado (amazon.titan-embed-text-v2:0).");
        console.log("- Modelo de chat habilitado (Anthropic) na mesma região.");
        console.log("- Política IAM com bedrock:InvokeModel.");
      }
    }
  }

  rl.close();
}

main().catch((e) => {
  console.error("Falha inesperada:", e);
  process.exit(1);
});
