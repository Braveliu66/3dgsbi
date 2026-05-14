// 本文件为 3DGS 预览系统内置 Spark 转码入口。
// 关键实现参考 https://github.com/sparkjsdev/spark 的 transcodeSpz/SpzReader。

import { readFile, writeFile } from "node:fs/promises";
import { SpzReader, transcodeSpz } from "@sparkjsdev/spark";

const DEFAULT_MAX_SH = 3;
const DEFAULT_FRACTIONAL_BITS = 14;

function readIntEnv(name, fallback, min, max) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

async function main() {
  const [command, inputPath, outputPath] = process.argv.slice(2);
  if (command === "convert") {
    if (!inputPath || !outputPath) throw new Error("usage: convert input.ply output.spz");
    const fileBytes = new Uint8Array(await readFile(inputPath));
    const maxSh = readIntEnv("SPARK_SPZ_MAX_SH", DEFAULT_MAX_SH, 0, 3);
    const fractionalBits = readIntEnv("SPARK_SPZ_FRACTIONAL_BITS", DEFAULT_FRACTIONAL_BITS, 8, 16);
    const result = await transcodeSpz({
      inputs: [{ fileBytes, pathOrUrl: inputPath }],
      maxSh,
      fractionalBits,
    });
    await writeFile(outputPath, result.fileBytes);
    const reader = new SpzReader({ fileBytes: result.fileBytes });
    await reader.parseHeader();
    console.log(JSON.stringify({ splats: reader.numSplats, bytes: result.fileBytes.byteLength, maxSh, fractionalBits }));
    return;
  }
  if (command === "validate") {
    if (!inputPath) throw new Error("usage: validate input.spz");
    const fileBytes = new Uint8Array(await readFile(inputPath));
    const reader = new SpzReader({ fileBytes });
    await reader.parseHeader();
    console.log(JSON.stringify({ splats: reader.numSplats }));
    return;
  }
  throw new Error("unknown command");
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});

