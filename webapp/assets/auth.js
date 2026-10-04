// JWT management, Google Sign-In mount, and a fetchJson helper that
// attaches the bearer token.

import { CONFIG } from "./config.js";

const TOKEN_KEY = "roadguard.jwt";

let apiOrigin = CONFIG.API_ORIGIN;
let originResolved = false;

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
