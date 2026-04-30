// Shared client helpers (kept intentionally tiny).
// Individual pages define their own logic; this file mainly avoids a missing-asset 404.

window.safeJson = async function safeJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
};

