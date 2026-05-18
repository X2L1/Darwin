"""Small built-in web UI for Darwin."""

from __future__ import annotations


def render_chat_ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Darwin</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #15191f;
      --muted: #5f6b7a;
      --line: #d8dee7;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --warn: #8a5a00;
      --soft: #eef7f6;
      --shadow: 0 12px 32px rgba(21, 25, 31, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      padding-bottom: 118px;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
    }

    header {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 18px 0 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    h1, h2 {
      margin: 0;
      font-weight: 700;
      line-height: 1.15;
    }

    h1 { font-size: 26px; }
    h2 { font-size: 15px; }
    a { color: var(--accent-strong); text-decoration: none; }
    a:hover { text-decoration: underline; }

    .subtle {
      color: var(--muted);
      font-size: 13px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .chat {
      min-height: calc(100vh - 132px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
    }

    .chat-top,
    .side-section {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }

    .messages {
      padding: 16px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bubble {
      max-width: min(760px, 92%);
      padding: 12px 14px;
      border-radius: 8px;
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 15px;
    }

    .bubble.user {
      align-self: flex-end;
      background: #1f2937;
      color: white;
    }

    .bubble.ai {
      align-self: flex-start;
      background: var(--soft);
      border: 1px solid #c6e5e0;
    }

    .composer {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    textarea,
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: white;
      outline: none;
    }

    textarea {
      min-height: 58px;
      max-height: 180px;
      resize: vertical;
    }

    textarea:focus,
    input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }

    button {
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 12px;
      background: white;
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }

    button:hover { border-color: var(--accent); }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      min-width: 86px;
    }
    button.primary:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }

    .quick {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }

    aside {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .side-body {
      padding: 14px;
      display: grid;
      gap: 10px;
    }

    .stat-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }

    .stat strong {
      display: block;
      font-size: 18px;
      margin-bottom: 2px;
    }

    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }

    .row input { min-width: 0; }

    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .toggle input {
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }

    .activity {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      min-height: 20px;
    }

    .warn {
      color: var(--warn);
    }

    details {
      font-size: 13px;
      color: var(--muted);
    }

    pre {
      max-height: 220px;
      overflow: auto;
      padding: 10px;
      border-radius: 8px;
      background: #f0f3f7;
      color: #1f2937;
      white-space: pre-wrap;
    }

    .training-dock {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 20;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 -12px 28px rgba(21, 25, 31, 0.08);
      backdrop-filter: blur(10px);
    }

    .training-inner {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 12px 0;
      display: grid;
      grid-template-columns: 210px minmax(0, 1fr);
      gap: 14px;
      align-items: center;
    }

    .training-score {
      display: grid;
      gap: 3px;
    }

    .training-score strong {
      font-size: 22px;
      line-height: 1;
    }

    .progress-track {
      height: 10px;
      border-radius: 999px;
      background: #e5eaf0;
      overflow: hidden;
      border: 1px solid var(--line);
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #0f766e, #2563eb);
      transition: width 240ms ease;
    }

    .milestones {
      margin-top: 9px;
      display: grid;
      grid-template-columns: repeat(8, minmax(86px, 1fr));
      gap: 6px;
    }

    .milestone {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 8px;
      background: #fbfcfd;
      display: grid;
      gap: 2px;
    }

    .milestone-label {
      font-size: 12px;
      font-weight: 700;
      line-height: 1.15;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .milestone-state {
      color: var(--muted);
      font-size: 11px;
      text-transform: capitalize;
    }

    .milestone.ready {
      border-color: #94d3b6;
      background: #eef8f2;
    }

    .milestone.learning {
      border-color: #9ac4ff;
      background: #eef5ff;
    }

    .milestone.started {
      border-color: #ead18a;
      background: #fff9e8;
    }

    .milestone.locked {
      opacity: 0.72;
    }

    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
      }
      body {
        padding-bottom: 214px;
      }
      .chat {
        min-height: 68vh;
      }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .training-inner {
        grid-template-columns: 1fr;
      }
      .milestones {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Darwin</h1>
      <div class="subtle">Local custom model with one chat surface</div>
    </div>
    <div class="subtle"><a href="/docs">API docs</a></div>
  </header>

  <main>
    <section class="panel chat" aria-label="Darwin chat">
      <div class="chat-top">
        <h2>Chat</h2>
        <div class="quick" aria-label="Quick actions">
          <button data-prompt="status">Status</button>
          <button data-prompt="show me the model source">Model source</button>
          <button data-prompt="improve yourself">Improve</button>
          <button data-prompt="search knowledge clean code">Search knowledge</button>
        </div>
      </div>

      <div id="messages" class="messages" aria-live="polite">
        <div class="bubble ai">Hi. I am Darwin. Ask me something, or use one of the buttons above.</div>
      </div>

      <form id="chat-form" class="composer">
        <textarea id="message" placeholder="Talk to Darwin..." autocomplete="off"></textarea>
        <button id="send" class="primary" type="submit">Send</button>
      </form>
    </section>

    <aside>
      <section class="panel">
        <div class="side-section">
          <h2>System</h2>
        </div>
        <div class="side-body">
          <div class="stat-grid">
            <div class="stat">
              <strong id="model-params">-</strong>
              <span class="subtle">parameters</span>
            </div>
            <div class="stat">
              <strong id="kb-count">-</strong>
              <span class="subtle">references</span>
            </div>
          </div>
          <button id="refresh-status" type="button">Refresh</button>
          <label class="toggle">
            <input id="run-improvements" type="checkbox" checked>
            Allow improvement cycles
          </label>
        </div>
      </section>

      <section class="panel">
        <div class="side-section">
          <h2>References</h2>
        </div>
        <div class="side-body">
          <input id="reference-path" placeholder="C:\\path\\to\\file-or-folder">
          <button id="add-reference" type="button">Add reference</button>
        </div>
      </section>

      <section class="panel">
        <div class="side-section">
          <h2>Activity</h2>
        </div>
        <div class="side-body">
          <div id="activity" class="activity">Ready.</div>
        </div>
      </section>
    </aside>
  </main>

  <section class="training-dock" aria-label="Training progress">
    <div class="training-inner">
      <div class="training-score">
        <strong id="training-percent">-</strong>
        <span id="training-level" class="subtle">Checking training...</span>
      </div>
      <div>
        <div class="progress-track" aria-hidden="true">
          <div id="training-fill" class="progress-fill"></div>
        </div>
        <div id="training-summary" class="subtle">Loading model milestones.</div>
        <div id="milestones" class="milestones"></div>
      </div>
    </div>
  </section>

  <script>
    const messages = document.getElementById("messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message");
    const sendButton = document.getElementById("send");
    const activity = document.getElementById("activity");
    const modelParams = document.getElementById("model-params");
    const kbCount = document.getElementById("kb-count");
    const runImprovements = document.getElementById("run-improvements");
    const trainingPercent = document.getElementById("training-percent");
    const trainingLevel = document.getElementById("training-level");
    const trainingFill = document.getElementById("training-fill");
    const trainingSummary = document.getElementById("training-summary");
    const milestones = document.getElementById("milestones");

    function setActivity(text, isWarning = false) {
      activity.textContent = text;
      activity.classList.toggle("warn", isWarning);
    }

    function addBubble(text, kind, details = null) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${kind}`;
      bubble.textContent = text;
      if (details) {
        const detail = document.createElement("details");
        const summary = document.createElement("summary");
        const pre = document.createElement("pre");
        summary.textContent = "Details";
        pre.textContent = JSON.stringify(details, null, 2);
        detail.append(summary, pre);
        bubble.append(document.createElement("br"), detail);
      }
      messages.appendChild(bubble);
      messages.scrollTop = messages.scrollHeight;
    }

    async function refreshStatus() {
      try {
        const response = await fetch("/status");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        modelParams.textContent = Number(data.model_parameters).toLocaleString();
        kbCount.textContent = Number(data.kb_entries).toLocaleString();
        setActivity("Status refreshed.");
      } catch (error) {
        setActivity(`Status failed: ${error.message}`, true);
      }
    }

    async function refreshTrainingProgress() {
      try {
        const response = await fetch("/training/progress");
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        trainingPercent.textContent = `${data.overall_percent}%`;
        trainingLevel.textContent = data.level;
        trainingFill.style.width = `${data.overall_percent}%`;
        trainingSummary.textContent = data.summary;
        milestones.replaceChildren(
          ...data.milestones.map((item) => {
            const node = document.createElement("div");
            node.className = `milestone ${item.status}`;
            node.title = `${item.label}: ${item.score}% - ${item.description}`;
            const label = document.createElement("div");
            label.className = "milestone-label";
            label.textContent = item.label;
            const state = document.createElement("div");
            state.className = "milestone-state";
            state.textContent = `${item.status} · ${item.score}%`;
            node.append(label, state);
            return node;
          })
        );
      } catch (error) {
        trainingPercent.textContent = "-";
        trainingLevel.textContent = "Unavailable";
        trainingSummary.textContent = `Training progress failed: ${error.message}`;
        milestones.replaceChildren();
      }
    }

    async function sendMessage(message) {
      const text = message.trim();
      if (!text) return;
      addBubble(text, "user");
      input.value = "";
      sendButton.disabled = true;
      setActivity("Darwin is thinking...");
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            max_new_tokens: 160,
            temperature: 0.8,
            run_improvements: runImprovements.checked
          })
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        addBubble(data.message, "ai", { intent: data.intent, actions: data.actions, data: data.data });
        const usedGroundedReply = data.actions.some(
          (action) => action.type === "grounded_system_reply"
        );
        await refreshStatus();
        setActivity(
          usedGroundedReply
            ? `Last intent: ${data.intent}; grounded reply used while the model keeps training.`
            : `Last intent: ${data.intent}`
        );
        await refreshTrainingProgress();
      } catch (error) {
        addBubble(`Something went wrong: ${error.message}`, "ai");
        setActivity("Request failed.", true);
      } finally {
        sendButton.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(input.value);
    });

    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => sendMessage(button.dataset.prompt));
    });

    document.getElementById("refresh-status").addEventListener("click", refreshStatus);

    document.getElementById("add-reference").addEventListener("click", async () => {
      const path = document.getElementById("reference-path").value.trim();
      if (!path) return;
      setActivity("Adding reference...");
      try {
        const response = await fetch("/knowledge/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path, tags: [], language: "en", is_primary_reference: true })
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        setActivity(`Added ${data.ingested} reference item(s).`);
        await refreshStatus();
        await refreshTrainingProgress();
      } catch (error) {
        setActivity(`Reference failed: ${error.message}`, true);
      }
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(input.value);
      }
    });

    refreshStatus();
    refreshTrainingProgress();
  </script>
</body>
</html>"""
