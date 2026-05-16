export type DetectedSceneType = "indoor" | "outdoor" | "auto";

export interface SceneHints {
  sceneType: DetectedSceneType;
  confidence: number;
  reasons: string[];
}

export async function detectSceneType(imageFiles: File[]): Promise<SceneHints> {
  const samples = imageFiles.filter((file) => file.type.startsWith("image/")).slice(0, 5);
  if (samples.length === 0) {
    return { sceneType: "auto", confidence: 0, reasons: ["no image samples"] };
  }

  const skyScores: number[] = [];
  for (const file of samples) {
    const canvas = await loadImage(file);
    skyScores.push(estimateSkyRatio(canvas));
  }

  const avgSkyScore = skyScores.reduce((sum, score) => sum + score, 0) / skyScores.length;
  if (avgSkyScore > 0.15) {
    return {
      sceneType: "outdoor",
      confidence: 0.8,
      reasons: [`sky ratio ${(avgSkyScore * 100).toFixed(1)}%`]
    };
  }

  return {
    sceneType: "indoor",
    confidence: 0.75,
    reasons: [`low sky ratio ${(avgSkyScore * 100).toFixed(1)}%`]
  };
}

function estimateSkyRatio(canvas: HTMLCanvasElement): number {
  const ctx = canvas.getContext("2d");
  if (!ctx) return 0;

  const width = canvas.width;
  const height = Math.max(1, Math.floor(canvas.height * 0.25));
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  let skyPixels = 0;
  const total = data.length / 4;

  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const brightness = (r + g + b) / 3;
    const bluish = b > r * 1.1 && b > g * 1.05;
    if (brightness > 160 && (bluish || brightness > 200)) {
      skyPixels += 1;
    }
  }

  return total > 0 ? skyPixels / total : 0;
}

function loadImage(file: File): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 200 / Math.max(image.width, image.height));
      canvas.width = Math.max(1, Math.round(image.width * scale));
      canvas.height = Math.max(1, Math.round(image.height * scale));
      canvas.getContext("2d")?.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Failed to load image sample: ${file.name}`));
    };
    image.src = url;
  });
}
