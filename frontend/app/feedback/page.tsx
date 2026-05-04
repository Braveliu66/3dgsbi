"use client";

import { Send } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";

export default function FeedbackPage() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setMessage(null);
    setError(null);
    try {
      await api.feedback({ title, content });
      setTitle("");
      setContent("");
      setMessage("反馈已记录，我们会尽快处理。");
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交失败");
    }
  }

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Feedback</p>
          <h1>问题反馈</h1>
        </div>
      </header>

      <section className="panel fill">
        <div className="panel-body scrollable stack">
          <div className="field">
            <label>标题</label>
            <input className="input" value={title} onChange={(event) => setTitle(event.target.value)} />
          </div>
          <div className="field" style={{ minHeight: 0 }}>
            <label>描述</label>
            <textarea className="textarea" value={content} onChange={(event) => setContent(event.target.value)} />
          </div>
          {message ? <div className="notice-box">{message}</div> : null}
          {error ? <div className="error-box">{error}</div> : null}
        </div>
        <div className="sticky-actions">
          <button className="button" type="button" onClick={() => void submit()} disabled={!title || !content}>
            <Send size={17} />提交反馈
          </button>
        </div>
      </section>
    </div>
  );
}
