"use client";

import Link from "next/link";
import { Camera, Images } from "lucide-react";

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
          <Link className="workflow-card simple" href="/camera">
            <span className="workflow-icon"><Camera size={30} /></span>
            <h2>实时流式扫描</h2>
            <p className="muted">实时相机管线暂未接入，当前页面只显示真实不可用状态。</p>
          </Link>
        </div>
      </section>
    </div>
  );
}
