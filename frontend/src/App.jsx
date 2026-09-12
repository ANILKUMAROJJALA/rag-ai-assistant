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


function getErrorMessage(
  error,
) {
  return (
    error?.message
    || "Something went wrong."
  );
}


async function parseResponse(
  response,
) {
  if (!response.ok) {

    let message =
      "Request failed.";

    try {

      const data =
        await response.json();

      if (data.detail) {
        message =
          data.detail;
      }

    } catch {
      // Keep fallback.
    }

    throw new Error(
      message
    );
  }

  return response.json();
}


function formatBytes(
  bytes,
) {
  if (
    bytes === 0
  ) {
    return "0 B";
  }

  if (!bytes) {
    return "";
  }

  const units = [
    "B",
    "KB",
    "MB",
    "GB",
  ];

  const index =
    Math.min(
      Math.floor(
        Math.log(bytes)
        / Math.log(1024)
      ),
      units.length - 1
    );

  const value =
    bytes
    / Math.pow(
      1024,
      index
    );

  return (
    `${value.toFixed(
      index === 0
        ? 0
        : 1
    )} ${units[index]}`
  );
}


function App() {

  const [
    conversations,
    setConversations,
  ] = useState([]);

  const [
    activeConversationId,
    setActiveConversationId,
  ] = useState(null);

  const [
    messages,
    setMessages,
  ] = useState([]);

  const [
    input,
    setInput,
  ] = useState("");

  const [
    loading,
    setLoading,
  ] = useState(false);

  const [
    pageLoading,
    setPageLoading,
  ] = useState(true);

  const [
    error,
    setError,
  ] = useState("");

  const [
    backendConnected,
    setBackendConnected,
  ] = useState(false);

  const [
    searchText,
    setSearchText,
  ] = useState("");

  const [
    sidebarOpen,
    setSidebarOpen,
  ] = useState(true);

  const [
    documents,
    setDocuments,
  ] = useState([]);

  const [
    documentPanelOpen,
    setDocumentPanelOpen,
  ] = useState(false);

  const [
    uploading,
    setUploading,
  ] = useState(false);

  const [
    uploadStatus,
    setUploadStatus,
  ] = useState("");

  const [
    showThemePanel,
    setShowThemePanel,
  ] = useState(false);

  const [
    accentColor,
    setAccentColor,
  ] = useState(() => {

    return (
      localStorage.getItem(
        "rag-accent-color"
      )
      || "#2563eb"
    );
  });


  const bottomRef =
    useRef(null);

  const fileInputRef =
    useRef(null);

  const themePanelRef =
    useRef(null);


  useEffect(() => {

    localStorage.setItem(
      "rag-accent-color",
      accentColor
    );

    document.documentElement
      .style
      .setProperty(
        "--accent",
        accentColor
      );

  }, [accentColor]);


  useEffect(() => {

    bottomRef.current
      ?.scrollIntoView({
        behavior: "smooth",
      });

  }, [
    messages,
    loading,
    error,
  ]);


  useEffect(() => {

    function handleOutsideClick(
      event,
    ) {
      if (
        themePanelRef.current
        && !themePanelRef.current
          .contains(
            event.target
          )
      ) {
        setShowThemePanel(
          false
        );
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


  useEffect(() => {

    initializeApp();

  }, []);


  async function initializeApp() {

    setPageLoading(true);

    await checkHealth();

    try {

      await Promise.all([
        refreshConversations(),
        refreshDocuments(),
      ]);

    } finally {

      setPageLoading(false);
    }
  }


  async function checkHealth() {

    try {

      const response =
        await fetch(
          `${API_URL}/health`
        );

      setBackendConnected(
        response.ok
      );

    } catch {

      setBackendConnected(
        false
      );
    }
  }


  async function refreshConversations() {

    try {

      const response =
        await fetch(
          `${API_URL}/conversations`
        );

      const data =
        await parseResponse(
          response
        );

      setConversations(
        data
      );

      return data;

    } catch (
      requestError
    ) {

      setBackendConnected(
        false
      );

      setError(
        getErrorMessage(
          requestError
        )
      );

      return [];
    }
  }


  async function refreshDocuments() {

    try {

      const response =
        await fetch(
          `${API_URL}/documents`
        );

      const data =
        await parseResponse(
          response
        );

      setDocuments(
        data
      );

      return data;

    } catch (
      requestError
    ) {

      console.error(
        "Could not load documents:",
        requestError
      );

      return [];
    }
  }


  async function createNewChat() {

    if (loading) {
      return;
    }

    setError("");

    try {

      const response =
        await fetch(
          `${API_URL}/conversations`,
          {
            method: "POST",
          }
        );

      const conversation =
        await parseResponse(
          response
        );

      setConversations(
        (current) => [
          conversation,
          ...current,
        ]
      );

      setActiveConversationId(
        conversation.id
      );

      setMessages([]);

      setInput("");

      setDocumentPanelOpen(
        false
      );

      setBackendConnected(
        true
      );

    } catch (
      requestError
    ) {

      setError(
        getErrorMessage(
          requestError
        )
      );
    }
  }


  async function openConversation(
    conversationId,
  ) {

    if (
      conversationId
      === activeConversationId
    ) {
      setDocumentPanelOpen(
        false
      );

      return;
    }

    setError("");
    setLoading(false);
    setDocumentPanelOpen(
      false
    );

    try {

      const response =
        await fetch(
          `${API_URL}/conversations/${conversationId}`
        );

      const conversation =
        await parseResponse(
          response
        );

      const loadedMessages =
        conversation.messages.map(
          (message) => ({
            id: message.id,
            role: message.role,
            content:
              message.content,
            sources:
              message.sources
              || [],
            route:
              message.route,
            retrievalRelevant:
              message
                .retrieval_relevant,
          })
        );

      setActiveConversationId(
        conversation.id
      );

      setMessages(
        loadedMessages
      );

      setBackendConnected(
        true
      );

    } catch (
      requestError
    ) {

      setError(
        getErrorMessage(
          requestError
        )
      );
    }
  }


  async function deleteChat(
    event,
    conversationId,
  ) {

    event.stopPropagation();

    const confirmed =
      window.confirm(
        "Delete this conversation?"
      );

    if (!confirmed) {
      return;
    }

    try {

      const response =
        await fetch(
          `${API_URL}/conversations/${conversationId}`,
          {
            method: "DELETE",
          }
        );

      await parseResponse(
        response
      );

      setConversations(
        (current) =>
          current.filter(
            (conversation) =>
              conversation.id
              !== conversationId
          )
      );

      if (
        activeConversationId
        === conversationId
      ) {
        setActiveConversationId(
          null
        );

        setMessages([]);
      }

    } catch (
      requestError
    ) {

      setError(
        getErrorMessage(
          requestError
        )
      );
    }
  }


  async function ensureConversation() {

    if (
      activeConversationId
    ) {
      return (
        activeConversationId
      );
    }

    const response =
      await fetch(
        `${API_URL}/conversations`,
        {
          method: "POST",
        }
      );

    const conversation =
      await parseResponse(
        response
      );

    setActiveConversationId(
      conversation.id
    );

    setConversations(
      (current) => [
        conversation,
        ...current,
      ]
    );

    return conversation.id;
  }


  async function handleSend() {

    const trimmedInput =
      input.trim();

    if (
      !trimmedInput
      || loading
    ) {
      return;
    }

    setError("");

    let conversationId;

    try {

      conversationId =
        await ensureConversation();

    } catch (
      requestError
    ) {

      setError(
        getErrorMessage(
          requestError
        )
      );

      return;
    }

    const userMessage = {
      id:
        crypto.randomUUID(),

      role:
        "user",

      content:
        trimmedInput,

      sources: [],
    };

    setMessages(
      (current) => [
        ...current,
        userMessage,
      ]
    );

    setInput("");
    setLoading(true);

    try {

      const response =
        await fetch(
          `${API_URL}/chat`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify({
                question:
                  trimmedInput,

                thread_id:
                  conversationId,
              }),
          }
        );

      const data =
        await parseResponse(
          response
        );

      if (!data.answer) {

        throw new Error(
          (
            "The server returned "
            + "an invalid response."
          )
        );
      }

      const assistantMessage = {
        id:
          crypto.randomUUID(),

        role:
          "assistant",

        content:
          data.answer,

        sources:
          data.sources
          || [],

        route:
          data.route,

        retrievalRelevant:
          data
            .retrieval_relevant,
      };

      setMessages(
        (current) => [
          ...current,
          assistantMessage,
        ]
      );

      setBackendConnected(
        true
      );

      await refreshConversations();

    } catch (
      requestError
    ) {

      console.error(
        "Chat request failed:",
        requestError
      );

      setError(
        getErrorMessage(
          requestError
        )
      );

    } finally {

      setLoading(
        false
      );
    }
  }


  function handleKeyDown(
    event,
  ) {

    if (
      event.key
      === "Enter"
      && !event.shiftKey
      && !loading
    ) {

      event.preventDefault();

      handleSend();
    }
  }


  function openFilePicker() {

    fileInputRef.current
      ?.click();
  }


  async function handleFileChange(
    event,
  ) {

    const file =
      event.target.files?.[0];

    event.target.value = "";

    if (!file) {
      return;
    }

    setUploading(true);

    setUploadStatus(
      `Uploading ${file.name}...`
    );

    setError("");

    try {

      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      const response =
        await fetch(
          `${API_URL}/documents/upload`,
          {
            method: "POST",
            body: formData,
          }
        );

      const documentInfo =
        await parseResponse(
          response
        );

      setUploadStatus(
        (
          `${documentInfo.name} `
          + `indexed successfully `
          + `(${documentInfo.chunks} chunks).`
        )
      );

      await refreshDocuments();

    } catch (
      requestError
    ) {

      setUploadStatus("");

      setError(
        getErrorMessage(
          requestError
        )
      );

    } finally {

      setUploading(false);
    }
  }


  async function handleDeleteDocument(
    documentName,
  ) {

    const confirmed =
      window.confirm(
        (
          `Delete ${documentName} `
          + "from the knowledge base?"
        )
      );

    if (!confirmed) {
      return;
    }

    try {

      const response =
        await fetch(
          (
            `${API_URL}/documents/`
            + encodeURIComponent(
              documentName
            )
          ),
          {
            method: "DELETE",
          }
        );

      await parseResponse(
        response
      );

      await refreshDocuments();

    } catch (
      requestError
    ) {

      setError(
        getErrorMessage(
          requestError
        )
      );
    }
  }


  const filteredConversations =
    conversations.filter(
      (conversation) =>
        conversation.title
          .toLowerCase()
          .includes(
            searchText
              .trim()
              .toLowerCase()
          )
    );


  return (
    <div className="app-shell">

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.txt"
        className="hidden-file-input"
        onChange={
          handleFileChange
        }
      />

      <aside
        className={
          sidebarOpen
            ? "sidebar"
            : "sidebar collapsed"
        }
      >

        <div className="sidebar-top">

          <div className="sidebar-brand">

            <div className="brand-mark">
              AI
            </div>

            {sidebarOpen && (
              <span>
                RAG Assistant
              </span>
            )}

          </div>

          <button
            className="icon-button"
            onClick={() =>
              setSidebarOpen(
                (current) =>
                  !current
              )
            }
            title="Toggle sidebar"
          >
            ☰
          </button>

        </div>


        {sidebarOpen && (
          <>

            <button
              className="new-chat-sidebar"
              onClick={
                createNewChat
              }
            >
              <span>
                ＋
              </span>

              New chat
            </button>


            <div className="sidebar-search">

              <span>
                ⌕
              </span>

              <input
                value={
                  searchText
                }
                onChange={
                  (event) =>
                    setSearchText(
                      event
                        .target
                        .value
                    )
                }
                placeholder="Search chats"
              />

            </div>


            <div className="sidebar-section">

              <div className="sidebar-label">
                Knowledge
              </div>

              <button
                className={
                  documentPanelOpen
                    ? "sidebar-nav-item active"
                    : "sidebar-nav-item"
                }
                onClick={() =>
                  setDocumentPanelOpen(
                    true
                  )
                }
              >
                <span>
                  ▣
                </span>

                My documents

                <span className="sidebar-count">
                  {
                    documents.length
                  }
                </span>
              </button>

              <button
                className="sidebar-nav-item"
                onClick={
                  openFilePicker
                }
                disabled={
                  uploading
                }
              >
                <span>
                  ＋
                </span>

                {
                  uploading
                    ? "Uploading..."
                    : "Upload document"
                }
              </button>

            </div>


            <div className="sidebar-section chats-section">

              <div className="sidebar-label">
                Chats
              </div>

              <div className="conversation-list">

                {
                  filteredConversations
                    .length === 0
                  && (
                    <div className="empty-sidebar">
                      No conversations yet.
                    </div>
                  )
                }

                {
                  filteredConversations
                    .map(
                      (conversation) => (

                        <button
                          key={
                            conversation.id
                          }
                          className={
                            (
                              activeConversationId
                              === conversation.id
                              && !documentPanelOpen
                            )
                              ? "conversation-item active"
                              : "conversation-item"
                          }
                          onClick={() =>
                            openConversation(
                              conversation.id
                            )
                          }
                        >

                          <span className="conversation-title">
                            {
                              conversation.title
                            }
                          </span>

                          <span
                            className="conversation-delete"
                            onClick={
                              (event) =>
                                deleteChat(
                                  event,
                                  conversation.id
                                )
                            }
                            title="Delete chat"
                          >
                            ×
                          </span>

                        </button>
                      )
                    )
                }

              </div>

            </div>

          </>
        )}

      </aside>


      <main className="main-area">

        <header className="topbar">

          <div>

            <div className="topbar-title">
              {
                documentPanelOpen
                  ? "My Documents"
                  : (
                    conversations.find(
                      (conversation) =>
                        conversation.id
                        === activeConversationId
                    )?.title
                    || "RAG AI Assistant"
                  )
              }
            </div>

            <div className="topbar-subtitle">
              Private document intelligence
            </div>

          </div>


          <div className="topbar-actions">

            <div
              className={
                backendConnected
                  ? "status-pill connected"
                  : "status-pill disconnected"
              }
            >
              <span className="status-dot" />

              {
                backendConnected
                  ? "Connected"
                  : "Disconnected"
              }
            </div>


            <div
              className="theme-wrapper"
              ref={
                themePanelRef
              }
            >

              <button
                className="topbar-button"
                onClick={() =>
                  setShowThemePanel(
                    (current) =>
                      !current
                  )
                }
              >
                Theme
              </button>


              {
                showThemePanel
                && (
                  <div className="theme-panel">

                    <div className="theme-title">
                      Appearance
                    </div>

                    <div className="theme-description">
                      Choose your accent color
                    </div>


                    <div className="theme-options">

                      {
                        ACCENT_PRESETS.map(
                          (preset) => (

                            <button
                              key={
                                preset.value
                              }
                              className={
                                accentColor
                                === preset.value
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
                                  background:
                                    preset.value,
                                }}
                              />

                              {
                                preset.name
                              }

                            </button>
                          )
                        )
                      }

                    </div>


                    <div className="custom-theme-row">

                      <span>
                        Custom
                      </span>

                      <input
                        type="color"
                        value={
                          accentColor
                        }
                        onChange={
                          (event) =>
                            setAccentColor(
                              event
                                .target
                                .value
                            )
                        }
                      />

                    </div>


                    <button
                      className="reset-theme"
                      onClick={() =>
                        setAccentColor(
                          "#2563eb"
                        )
                      }
                    >
                      Reset to default
                    </button>

                  </div>
                )
              }

            </div>

          </div>

        </header>


        {
          documentPanelOpen
            ? (
              <section className="documents-page">

                <div className="documents-header">

                  <div>

                    <h1>
                      Knowledge base
                    </h1>

                    <p>
                      Upload PDF, DOCX,
                      or TXT files.
                      Indexed documents
                      become immediately
                      available to the
                      RAG assistant.
                    </p>

                  </div>

                  <button
                    className="primary-button"
                    onClick={
                      openFilePicker
                    }
                    disabled={
                      uploading
                    }
                  >
                    {
                      uploading
                        ? "Uploading..."
                        : "+ Upload document"
                    }
                  </button>

                </div>


                {
                  uploadStatus
                  && (
                    <div className="success-banner">
                      {
                        uploadStatus
                      }
                    </div>
                  )
                }


                <div className="document-grid">

                  {
                    documents.length
                    === 0
                    && (
                      <div className="empty-documents">
                        <div className="empty-documents-icon">
                          ▣
                        </div>

                        <h3>
                          No documents yet
                        </h3>

                        <p>
                          Upload your first
                          private document
                          to start building
                          the knowledge base.
                        </p>
                      </div>
                    )
                  }


                  {
                    documents.map(
                      (document) => (

                        <div
                          className="document-card"
                          key={
                            document.name
                          }
                        >

                          <div className="document-icon">
                            {
                              document.file_type
                                .toUpperCase()
                            }
                          </div>

                          <div className="document-info">

                            <div className="document-name">
                              {
                                document.name
                              }
                            </div>

                            <div className="document-meta">
                              {
                                document.file_type
                                  .toUpperCase()
                              }
                              {" · "}
                              {
                                formatBytes(
                                  document.size
                                )
                              }
                            </div>

                            <div className="document-status">
                              <span />
                              Ready
                            </div>

                          </div>

                          <button
                            className="delete-document-button"
                            onClick={() =>
                              handleDeleteDocument(
                                document.name
                              )
                            }
                            title="Delete document"
                          >
                            Delete
                          </button>

                        </div>
                      )
                    )
                  }

                </div>

              </section>
            )
            : (
              <section className="chat-layout">

                <div className="chat-scroll">

                  {
                    pageLoading
                    && (
                      <div className="center-state">
                        Loading...
                      </div>
                    )
                  }


                  {
                    !pageLoading
                    && messages.length
                    === 0
                    && (
                      <div className="welcome-state">

                        <div className="welcome-logo">
                          AI
                        </div>

                        <h1>
                          What can I help
                          you find?
                        </h1>

                        <p>
                          Ask questions
                          grounded in your
                          private document
                          knowledge base.
                        </p>

                        <div className="welcome-actions">

                          <button
                            onClick={
                              openFilePicker
                            }
                          >
                            Upload a document
                          </button>

                          <button
                            onClick={() =>
                              setDocumentPanelOpen(
                                true
                              )
                            }
                          >
                            View knowledge base
                          </button>

                        </div>

                      </div>
                    )
                  }


                  <div className="message-container">

                    {
                      messages.map(
                        (message) => (

                          <div
                            className={
                              message.role
                              === "user"
                                ? "message user-message"
                                : "message assistant-message"
                            }
                            key={
                              message.id
                            }
                          >

                            <div className="message-avatar">
                              {
                                message.role
                                === "user"
                                  ? "U"
                                  : "AI"
                              }
                            </div>


                            <div className="message-body">

                              <div className="message-role">
                                {
                                  message.role
                                  === "user"
                                    ? "You"
                                    : "Assistant"
                                }
                              </div>

                              <div className="message-text">
                                {
                                  message.content
                                }
                              </div>


                              {
                                message.sources
                                  ?.length > 0
                                && (
                                  <div className="sources">

                                    <div className="sources-title">
                                      Sources
                                    </div>

                                    <div className="source-chips">

                                      {
                                        message.sources
                                          .map(
                                            (
                                              source,
                                              index,
                                            ) => (

                                              <div
                                                className="source-chip"
                                                key={
                                                  (
                                                    source.source
                                                    || "source"
                                                  )
                                                  + index
                                                }
                                              >

                                                <span>
                                                  ▤
                                                </span>

                                                <span>
                                                  {
                                                    source.source
                                                    || "Unknown source"
                                                  }

                                                  {
                                                    source.page
                                                    ? (
                                                      ` · p.${source.page}`
                                                    )
                                                    : ""
                                                  }
                                                </span>

                                              </div>
                                            )
                                          )
                                      }

                                    </div>

                                  </div>
                                )
                              }

                            </div>

                          </div>
                        )
                      )
                    }


                    {
                      loading
                      && (
                        <div className="message assistant-message">

                          <div className="message-avatar">
                            AI
                          </div>

                          <div className="message-body">

                            <div className="message-role">
                              Assistant
                            </div>

                            <div className="thinking-row">

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
                      )
                    }


                    {
                      error
                      && (
                        <div className="error-banner">

                          <strong>
                            Request failed
                          </strong>

                          <span>
                            {
                              error
                            }
                          </span>

                        </div>
                      )
                    }


                    <div
                      ref={
                        bottomRef
                      }
                    />

                  </div>

                </div>


                <div className="composer-area">

                  <div className="composer-box">

                    <button
                      className="attach-button"
                      onClick={
                        openFilePicker
                      }
                      title="Upload document"
                    >
                      ＋
                    </button>

                    <textarea
                      value={
                        input
                      }
                      onChange={
                        (event) =>
                          setInput(
                            event
                              .target
                              .value
                          )
                      }
                      onKeyDown={
                        handleKeyDown
                      }
                      placeholder={
                        loading
                          ? "Waiting for response..."
                          : "Ask anything about your documents"
                      }
                      disabled={
                        loading
                      }
                      rows={1}
                    />

                    <button
                      className="send-button"
                      disabled={
                        loading
                        || !input.trim()
                      }
                      onClick={
                        handleSend
                      }
                    >
                      ↑
                    </button>

                  </div>

                  <div className="composer-note">
                    Responses are grounded
                    in indexed documents.
                    Verify important
                    information against
                    the cited source.
                  </div>

                </div>

              </section>
            )
        }

      </main>

    </div>
  );
}


export default App;