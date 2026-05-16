"use client";

import Link from "next/link";
import { MessageSquare, UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import { formatDateTime } from "@/lib/labels";
import type { User } from "@/lib/types";

export default function ProfilePage() {
  const [user, setUser] = useState<User | null>(null);
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([api.me(), api.projectSummary()])
      .then(([userData, summaryData]) => {
        setUser(userData);
        setSummary(summaryData);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "用户信息加载失败"));
  }, []);

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Profile</p>
          <h1>用户中心</h1>
        </div>
        <Link className="button" href="/feedback">
          <MessageSquare size={17} />反馈
        </Link>
      </header>

      <section className="detail-grid">
        <div className="panel padded profile-hero">
          <span className="profile-avatar">{user?.username.slice(0, 1).toUpperCase() || <UserRound size={30} />}</span>
          <div>
            <h2>{user?.username ?? "加载中"}</h2>
            <p className="muted small">
              {user?.role === "admin" ? "管理员" : "普通用户"} · 创建于 {formatDateTime(user?.created_at)}
            </p>
            {user?.email ? <p className="muted small">{user.email}</p> : null}
          </div>
        </div>

        <aside className="detail-side">
          <section className="grid two">
            <div className="panel stat"><span className="muted small">项目数</span><strong>{summary.total ?? 0}</strong></div>
            <div className="panel stat"><span className="muted small">运行中</span><strong>{summary.running ?? 0}</strong></div>
            <div className="panel stat"><span className="muted small">已完成</span><strong>{summary.completed ?? 0}</strong></div>
            <div className="panel stat"><span className="muted small">失败</span><strong>{summary.failed ?? 0}</strong></div>
          </section>
          <div className="panel stat">
            <span className="muted small">占用空间</span>
            <strong>{formatBytes(summary.total_size_bytes ?? 0)}</strong>
          </div>
          {error ? <div className="error-box">{error}</div> : null}
        </aside>
      </section>
    </div>
  );
}
