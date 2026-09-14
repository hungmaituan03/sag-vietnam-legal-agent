const thread = document.getElementById("thread");
const empty = document.getElementById("empty");
const form = document.getElementById("composer");
const input = document.getElementById("query");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");

function setStatus(text, isError = false) {
  if (!text) {
    statusEl.hidden = true;
    statusEl.textContent = "";
    statusEl.classList.remove("error");
    return;
  }
  statusEl.hidden = false;
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function appendMessage(role, html, className = "") {
  if (empty) empty.remove();
  const el = document.createElement("article");
  el.className = `msg ${role}${className ? ` ${className}` : ""}`;
  el.innerHTML = html;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
  return el;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function selectedMode() {
  const picked = form.querySelector('input[name="mode"]:checked');
  return picked ? picked.value : "sag";
}

function renderArm(arm) {
  const answer = escapeHtml(arm.answer || "");
  const stats = arm.stats || {};
  let html = `<div class="arm-label">${escapeHtml(arm.label || "")}</div>`;
  html += `<div class="body">${answer}</div>`;
  html += `<div class="meta">Evidence ${stats.context_count ?? "—"} chunks (seeds ${
    stats.seed_count ?? "—"
  }, +SAG ${stats.sag_added ?? "—"}) · bên dưới là căn cứ thô, không phải câu trả lời</div>`;

  if (Array.isArray(arm.cited) && arm.cited.length) {
    const items = arm.cited
      .map((c) => {
        const title = escapeHtml(c.title || c.document_id || "");
        const path = escapeHtml(c.citation_path || "");
        const text = escapeHtml(c.text || "");
        return `<div class="cite"><strong>${title}</strong> — ${path}<br />${text}</div>`;
      })
      .join("");
    html += `<details class="citations"><summary>Căn cứ (${arm.cited.length})</summary>${items}</details>`;
  }
  return html;
}

function renderAssistant(data) {
  const arms = Array.isArray(data.arms) ? data.arms : [];
  if (!arms.length) {
    appendMessage("assistant", "Không có kết quả.", "abstained");
    return;
  }

  if (arms.length === 1) {
    const arm = arms[0];
    appendMessage(
      "assistant",
      renderArm(arm),
      arm.abstained ? "abstained" : ""
    );
    return;
  }

  const panels = arms
    .map((arm) => {
      const cls = arm.use_sag ? "arm sag" : "arm rag";
      const abs = arm.abstained ? " abstained" : "";
      return `<div class="${cls}${abs}">${renderArm(arm)}</div>`;
    })
    .join("");
  appendMessage(
    "assistant",
    `<div class="compare-grid">${panels}</div>`,
    "compare"
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  const mode = selectedMode();
  const modeNote =
    mode === "compare"
      ? "so sánh RAG vs SAG"
      : mode === "rag"
        ? "RAG (no SAG)"
        : "SAG";

  appendMessage(
    "user",
    `${escapeHtml(query)}<div class="meta">Chế độ: ${escapeHtml(modeNote)}</div>`
  );
  input.value = "";
  send.disabled = true;
  setStatus(
    mode === "compare"
      ? "Đang chạy RAG rồi SAG để so sánh… (lần đầu có thể chậm)"
      : "Đang truy xuất và soạn trả lời… (lần đầu có thể chậm)"
  );

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode }),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = payload.detail || res.statusText || "Lỗi không xác định";
      appendMessage(
        "assistant",
        escapeHtml(typeof detail === "string" ? detail : JSON.stringify(detail)),
        "abstained"
      );
      setStatus("Không trả lời được.", true);
      return;
    }
    renderAssistant(payload);
    setStatus("");
  } catch (err) {
    appendMessage(
      "assistant",
      escapeHtml(err.message || "Không kết nối được máy chủ."),
      "abstained"
    );
    setStatus("Lỗi mạng.", true);
  } finally {
    send.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    if (h.status !== "ok") {
      const missing = [];
      if (!h.corpus) missing.push("corpus");
      if (!h.voyage) missing.push("VOYAGE_API_KEY");
      if (!h.openai) missing.push("OPENAI_API_KEY");
      setStatus(`Chưa sẵn sàng: thiếu ${missing.join(", ")}`, true);
    }
  })
  .catch(() => setStatus("Không đọc được /api/health", true));
