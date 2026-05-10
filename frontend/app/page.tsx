"use client";

import Link from "next/link";
import { Images } from "lucide-react";

export default function HomePage() {
  return (
    <div className="workspace-page home-choice-page">
      <section className="home-choice">
        <h1>选择重建工作流</h1>
        <div className="workflow-grid">
          <Link className="workflow-card simple" href="/upload">
            <span className="workflow-icon"><Images size={30} /></span>
            <h2>离线数据集重建</h2>
            <p className="muted">上传图片序列或视频，可直接训练，也可先生成极速预览。</p>
          </Link>
        </div>
      </section>
    </div>
  );
}
