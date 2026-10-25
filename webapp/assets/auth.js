// JWT management, Google Sign-In mount, and a fetchJson helper that
// attaches the bearer token.

import { CONFIG } from "./config.js";

const TOKEN_KEY = "roadguard.jwt";

let apiOrigin = CONFIG.API_ORIGIN;
let originResolved = false;
let unauthorizedHandler = null;

// app.js registers the sign-in modal here; this module cannot import it.
export function setUnauthorizedHandler(fn) {
  unauthorizedHandler = fn;
}

export async function resolveApiOrigin() {
  if (originResolved) return apiOrigin;
  try {
    const resp = await fetch(CONFIG.TUNNEL_URL_JSON, { cache: "no-store" });
    if (resp.ok) {
      const data = await resp.json();
      if (data && typeof data.origin === "string" && data.origin) {
        apiOrigin = data.origin.replace(/\/+$/, "");
      }
    }
  } catch {
    // keep the default origin
  }
  originResolved = true;
  return apiOrigin;
}

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function saveToken(token) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // storage unavailable
  }
}

export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore
  }
}

export function isSignedIn() {
  const t = getToken();
  if (!t) return false;
  const parts = t.split(".");
  if (parts.length !== 3) return false;
  try {
    const payload = JSON.parse(
      atob(parts[1].replace(/-/g, "+").replace(/_/g, "/"))
    );
    if (payload.exp && Date.now() / 1000 > payload.exp) {
      clearToken();
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

export function parseJwt() {
  const t = getToken();
  if (!t) return null;
  const parts = t.split(".");
  if (parts.length !== 3) return null;
  try {
    return JSON.parse(atob(parts[1].replace(/-/g, "+").replace(/_/g, "/")));
  } catch {
    return null;
  }
}

export async function fetchJson(path, options = {}) {
  const origin = await resolveApiOrigin();
  const url = `${origin}${path.startsWith("/") ? "" : "/"}${path}`;
  const headers = Object.assign(
    { "Content-Type": "application/json" },
    options.headers || {}
  );
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const resp = await fetch(url, { ...options, headers, credentials: "omit" });
  const contentType = resp.headers.get("Content-Type") || "";
  const payload = contentType.includes("application/json")
    ? await resp.json().catch(() => null)
    : await resp.text();
  if (!resp.ok) {
    const detail =
      payload && typeof payload === "object" ? payload.detail || payload : payload;
    const err = new Error(
      `http_${resp.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`
    );
    err.status = resp.status;
    err.payload = payload;
    // A rejected bearer token stays rejected: drop it so the next call
    // starts signed out. A wrong passphrase is a 401 too, hence the
    // login exclusion.
    if (resp.status === 401 && !/^\/?auth\/login$/.test(path)) {
      clearToken();
      if (unauthorizedHandler) unauthorizedHandler(err);
    }
    throw err;
  }
  return payload;
}

export async function loginWithPassphrase(passphrase) {
  const data = await fetchJson("/auth/login", {
    method: "POST",
    body: JSON.stringify({ passphrase }),
  });
  if (!data || !data.token) {
    throw new Error("login_response_missing_token");
  }
  saveToken(data.token);
  return data;
}

// Step 2 (25.10.2026): username + password login against the new endpoint.
// Rejects with a stable `invalid_credentials` message on wrong user/wrong
// password alike, so the UI must show the same text for both.
export async function loginWithUsernamePassword(username, password) {
  const data = await fetchJson("/auth/login/password", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  if (!data || !data.token) {
    throw new Error("login_response_missing_token");
  }
  saveToken(data.token);
  return data;
}

// Step 2 (25.10.2026): user registration. Gated on the server by
// ROADGUARD_SIGNUP_ENABLED; production stays off, so this call will 403.
// The UI only renders the signup tab after it has fetched /auth/config.
export async function registerAccount(username, password, email) {
  const body = { username, password };
  if (email) body.email = email;
  const data = await fetchJson("/auth/register", {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (!data || !data.token) {
    throw new Error("register_response_missing_token");
  }
  saveToken(data.token);
  return data;
}

// Step 2 (25.10.2026): the UI reads which login methods are enabled before
// deciding which tabs/buttons to render.
export async function fetchAuthConfig() {
  return fetchJson("/auth/config");
}

// Step 2 (25.10.2026): server-side logout. Stateless on the server
// (just returns {ok:true}); the client drops the token locally too.
export async function logoutCall() {
  try {
    await fetchJson("/auth/logout", { method: "POST" });
  } catch {
    // Even if the server call fails, continue - the client clears locally.
  }
  clearToken();
}

export async function loginWithGoogleIdToken(idToken) {
  const data = await fetchJson("/auth/login", {
    method: "POST",
    body: JSON.stringify({ google_id_token: idToken }),
  });
  if (!data || !data.token) {
    throw new Error("login_response_missing_token");
  }
  saveToken(data.token);
  return data;
}

export function mountGoogleSignIn(containerId, onCredential) {
  if (!CONFIG.GOOGLE_CLIENT_ID) return;
  const el = document.getElementById(containerId);
  if (!el) return;
  // Wait for Google's script to load (it defers).
  const tryMount = () => {
    if (!window.google || !window.google.accounts || !window.google.accounts.id) {
      setTimeout(tryMount, 150);
      return;
    }
    window.google.accounts.id.initialize({
      client_id: CONFIG.GOOGLE_CLIENT_ID,
      callback: (resp) => onCredential(resp.credential),
    });
    window.google.accounts.id.renderButton(el, {
      theme: "outline",
      size: "large",
      text: "signin_with",
      shape: "rectangular",
      logo_alignment: "left",
    });
  };
  tryMount();
}

export async function fetchMe() {
  return fetchJson("/auth/me");
}
