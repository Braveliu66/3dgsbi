// 本文件为 3DGS 预览系统内置 Spark 转码入口。
// 关键实现参考 https://github.com/sparkjsdev/spark 的 transcodeSpz/SpzReader。

import { readFile, writeFile } from "node:fs/promises";
import { SpzReader, transcodeSpz } from "@sparkjsdev/spark";

async function main() {
  const [command, inputPath, outputPath] = process.argv.slice(2);
  if (command === "convert") {
    if (!inputPath || !outputPath) throw new Error("usage: convert input.ply output.spz");
    const fileBytes = new Uint8Array(await readFile(inputPath));
    const result = await transcodeSpz({
      inputs: [{ fileBytes, pathOrUrl: inputPath }],
      maxSh: 0,
      fractionalBits: 12,
    });
    await writeFile(outputPath, result.fileBytes);
    const reader = new SpzReader({ fileBytes: result.fileBytes });
    await reader.parseHeader();
    console.log(JSON.stringify({ splats: reader.numSplats, bytes: result.fileBytes.byteLength }));
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

