/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    VIEWER_TARGET_FPS: process.env.VIEWER_TARGET_FPS ?? "90",
    VIEWER_QUALITY_UP_FPS: process.env.VIEWER_QUALITY_UP_FPS ?? "105",
    VIEWER_QUALITY_DOWN_FPS: process.env.VIEWER_QUALITY_DOWN_FPS ?? "90",
    VIEWER_ADAPTIVE_QUALITY: process.env.VIEWER_ADAPTIVE_QUALITY ?? "true"
  }
};

export default nextConfig;

