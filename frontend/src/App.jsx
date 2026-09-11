import {
  useEffect,
  useRef,
  useState,
} from "react";

import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL
  || "http://127.0.0.1:8000";

const ACCENT_PRESETS = [
  {
    name: "Blue",
    value: "#2563eb",
  },
  {
    name: "Purple",
    value: "#7c3aed",
  },
  {
    name: "Green",
    value: "#16a34a",
  },
  {
    name: "Orange",
    value: "#ea580c",
  },
  {
    name: "Pink",
    value: "#db2777",
  },
];


function createWelcomeMessage() {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    content:
      "Hi! I’m your RAG AI Assistant. Ask me a question about the documents in the knowledge base.",
    sources: [],
  };
}


function App() {

  // --------------------------------------------------
  // Chat state
  // --------------------------------------------------

  const [input, setInput] = useState("");

  const [messages, setMessages] = useState([
    createWelcomeMessage(),
  ]);

  const [threadId, setThreadId] = useState(
    () => crypto.randomUUID()
  );

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  // --------------------------------------------------
  // Theme state
  // --------------------------------------------------

  const [showThemePanel, setShowThemePanel] =
    useState(false);

  const [accentColor, setAccentColor] =
    useState(() => {

      const savedColor =
        localStorage.getItem(
          "rag-accent-color"
        );

      return savedColor || "#2563eb";
    });


  // --------------------------------------------------
  // Refs
  // --------------------------------------------------

  const bottomRef = useRef(null);

  const themePanelRef = useRef(null);


  // --------------------------------------------------
  // Apply and save accent color
  // --------------------------------------------------

  useEffect(() => {

    localStorage.setItem(
      "rag-accent-color",
      accentColor
    );

    document.documentElement.style.setProperty(
      "--accent",
      accentColor
    );

  }, [accentColor]);


  // --------------------------------------------------
  // Auto-scroll
  // --------------------------------------------------

  useEffect(() => {

    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });

  }, [messages, loading, error]);


  // --------------------------------------------------
  // Close theme panel when clicking outside
  // --------------------------------------------------

  useEffect(() => {

    function handleOutsideClick(event) {

      if (
        themePanelRef.current
        && !themePanelRef.current.contains(
          event.target
        )
      ) {

        setShowThemePanel(false);

      }
    }


    document.addEventListener(
      "mousedown",
      handleOutsideClick
    );


    return () => {

      document.removeEventListener(
        "mousedown",
        handleOutsideClick
      );

    };

  }, []);


  // --------------------------------------------------
  // Send message
  // --------------------------------------------------

  async function handleSend() {

    const trimmedInput = input.trim();

    if (!trimmedInput || loading) {
      return;
    }


    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedInput,
      sources: [],
    };


    setMessages((previousMessages) => [
      ...previousMessages,
      userMessage,
    ]);


    setInput("");

    setError("");

    setLoading(true);


    try {

      const response = await fetch(
            `${API_URL}/chat`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            question: trimmedInput,
            thread_id: threadId,
          }),
        }
      );


      if (!response.ok) {

        throw new Error(
          `Request failed with status ${response.status}`
        );

      }


      const data = await response.json();


      const assistantMessage = {
        id: crypto.randomUUID(),

        role: "assistant",

        content: data.answer,

        sources: data.sources || [],

        route: data.route,

        retrievalRelevant:
          data.retrieval_relevant,
      };


      setMessages((previousMessages) => [
        ...previousMessages,
        assistantMessage,
      ]);

    } catch (requestError) {

      console.error(
        "Chat request failed:",
        requestError
      );


      setError(
        "Unable to reach the RAG assistant. Make sure the FastAPI server is running."
      );

    } finally {

      setLoading(false);

    }
  }


  // --------------------------------------------------
  // Keyboard handling
  // --------------------------------------------------

  function handleKeyDown(event) {

    if (
      event.key === "Enter"
      && !event.shiftKey
      && !loading
    ) {

      event.preventDefault();

      handleSend();

    }
  }


  // --------------------------------------------------
  // New chat
  // --------------------------------------------------

  function handleNewChat() {

    setThreadId(
      crypto.randomUUID()
    );

    setMessages([
      createWelcomeMessage(),
    ]);

    setInput("");

    setError("");

    setLoading(false);
  }


  // --------------------------------------------------
  // Reset theme
  // --------------------------------------------------

  function handleResetTheme() {

    setAccentColor(
      "#2563eb"
    );
  }


  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div className="app-shell">

      {/* ------------------------------------------------
          Topbar
      ------------------------------------------------ */}

      <header className="topbar">

        <div className="brand-block">

          <div className="brand-icon">
            AI
          </div>


          <div>

            <h1>
              RAG AI Assistant
            </h1>

            <p>
              Private document intelligence
            </p>

          </div>

        </div>


        <div className="topbar-actions">

          <div className="status-pill">

            <span className="status-dot" />

            Backend connected

          </div>


          {/* Theme selector */}

          <div
            className="theme-wrapper"
            ref={themePanelRef}
          >

            <button
              className="theme-button"
              onClick={() =>
                setShowThemePanel(
                  (previousValue) =>
                    !previousValue
                )
              }
            >

              <span
                className="theme-button-color"
              />

              Theme

            </button>


            {showThemePanel && (

              <div className="theme-panel">

                <div className="theme-panel-header">

                  <div>

                    <div className="theme-title">
                      Appearance
                    </div>

                    <div className="theme-description">
                      Choose your accent color
                    </div>

                  </div>

                </div>


                <div className="theme-section-label">
                  Presets
                </div>


                <div className="theme-presets">

                  {ACCENT_PRESETS.map(
                    (preset) => (

                      <button
                        key={preset.value}

                        type="button"

                        className={
                          accentColor === preset.value
                            ? "theme-option active"
                            : "theme-option"
                        }

                        onClick={() =>
                          setAccentColor(
                            preset.value
                          )
                        }
                      >

                        <span
                          className="theme-swatch"
                          style={{
                            backgroundColor:
                              preset.value,
                          }}
                        />

                        <span>
                          {preset.name}
                        </span>

                      </button>

                    )
                  )}

                </div>


                <div className="custom-color-section">

                  <div>

                    <div className="custom-color-title">
                      Custom color
                    </div>

                    <div className="custom-color-description">
                      Pick any color you like
                    </div>

                  </div>


                  <label
                    className="color-picker-wrapper"
                  >

                    <span
                      className="selected-color-preview"
                      style={{
                        backgroundColor:
                          accentColor,
                      }}
                    />

                    <input
                      type="color"

                      value={accentColor}

                      onChange={(event) =>
                        setAccentColor(
                          event.target.value
                        )
                      }

                      aria-label="Choose custom accent color"
                    />

                  </label>

                </div>


                <div className="theme-value">

                  <span>
                    Current
                  </span>

                  <code>
                    {accentColor.toUpperCase()}
                  </code>

                </div>


                <button
                  className="reset-theme-button"
                  onClick={handleResetTheme}
                >
                  Reset to default
                </button>

              </div>

            )}

          </div>


          <button
            className="new-chat-button"
            onClick={handleNewChat}
          >
            New Chat
          </button>

        </div>

      </header>


      {/* ------------------------------------------------
          Main app
      ------------------------------------------------ */}

      <main className="main-layout">

        <section className="chat-panel">


          {/* Chat panel header */}

          <div className="chat-header">

            <div>

              <h2>
                Document Assistant
              </h2>

              <p>
                Grounded answers with source attribution
              </p>

            </div>


            <div className="rag-badge">
              RAG Enabled
            </div>

          </div>


          {/* ------------------------------------------------
              Messages
          ------------------------------------------------ */}

          <div className="chat-body">

            {messages.map((message) => (

              <div
                key={message.id}

                className={
                  message.role === "user"
                    ? "message-row user-row"
                    : "message-row assistant-row"
                }
              >

                <div className="avatar">

                  {message.role === "user"
                    ? "U"
                    : "AI"}

                </div>


                <div
                  className={
                    message.role === "user"
                      ? "message-card user-card"
                      : "message-card assistant-card"
                  }
                >

                  <div className="message-meta">

                    {message.role === "user"
                      ? "You"
                      : "Assistant"}

                  </div>


                  <div className="message-content">

                    {message.content}

                  </div>


                  {/* Sources */}

                  {message.sources.length > 0 && (

                    <div className="sources-section">

                      <div className="sources-heading">

                        <span>
                          Sources
                        </span>

                        <span className="sources-count">
                          {message.sources.length}
                        </span>

                      </div>


                      <div className="source-list">

                        {message.sources.map(
                          (source, index) => (

                            <div
                              key={
                                `${source.source}-${index}`
                              }

                              className="source-chip"
                            >

                              <span className="source-icon">
                                DOC
                              </span>


                              <span className="source-name">

                                {source.source
                                  || "Unknown source"}

                                {source.page
                                  ? ` · page ${source.page}`
                                  : ""}

                              </span>

                            </div>

                          )
                        )}

                      </div>

                    </div>

                  )}

                </div>

              </div>

            ))}


            {/* ------------------------------------------------
                Loading state
            ------------------------------------------------ */}

            {loading && (

              <div className="message-row assistant-row">

                <div className="avatar">
                  AI
                </div>


                <div className="message-card assistant-card">

                  <div className="message-meta">
                    Assistant
                  </div>


                  <div className="thinking">

                    <span>
                      Thinking
                    </span>


                    <div className="thinking-dots">

                      <span />
                      <span />
                      <span />

                    </div>

                  </div>

                </div>

              </div>

            )}


            {/* ------------------------------------------------
                Error
            ------------------------------------------------ */}

            {error && (

              <div className="error-card">

                <div className="error-title">
                  Connection error
                </div>

                <div>
                  {error}
                </div>

              </div>

            )}


            <div ref={bottomRef} />

          </div>


          {/* ------------------------------------------------
              Composer
          ------------------------------------------------ */}

          <div className="composer">

            <div className="composer-box">

              <textarea
                value={input}

                placeholder={
                  loading
                    ? "Waiting for response..."
                    : "Ask a question about your documents..."
                }

                onChange={(event) =>
                  setInput(
                    event.target.value
                  )
                }

                onKeyDown={handleKeyDown}

                disabled={loading}

                rows={1}
              />


              <button
                className="send-button"

                onClick={handleSend}

                disabled={
                  loading
                  || !input.trim()
                }
              >

                {loading
                  ? "..."
                  : "Send"}

              </button>

            </div>


            <div className="composer-footer">

              <span>
                Enter to send
              </span>

              <span className="footer-divider">
                •
              </span>

              <span>
                Shift + Enter for a new line
              </span>

            </div>

          </div>

        </section>

      </main>

    </div>
  );
}


export default App;