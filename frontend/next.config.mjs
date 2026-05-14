/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    proxyClientMaxBodySize: "64mb"
  },
  async rewrites() {
    const apiBase = process.env.INTERNAL_API_BASE_URL ?? "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiBase}/api/:path*`
      }
    ];
  },
  env: {
    VIEWER_TARGET_FPS: process.env.VIEWER_TARGET_FPS ?? "60",
    VIEWER_QUALITY_UP_FPS: process.env.VIEWER_QUALITY_UP_FPS ?? "72",
    VIEWER_QUALITY_DOWN_FPS: process.env.VIEWER_QUALITY_DOWN_FPS ?? "45",
    VIEWER_ADAPTIVE_QUALITY: process.env.VIEWER_ADAPTIVE_QUALITY ?? "true"
  }
};

export default nextConfig;

