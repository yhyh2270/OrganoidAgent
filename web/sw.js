self.skipWaiting();

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      caches.keys().then((keys) =>
        Promise.all(keys.filter((key) => key !== "organoid-agent-v24").map((key) => caches.delete(key)))
      ),
      self.clients.claim(),
    ])
  );
});

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("organoid-agent-v24").then((cache) =>
      cache.addAll([
        "/",
        "/static/styles.css?v=24",
        "/static/app.js?v=24",
        "/static/manifest.json",
        "/static/icons/icon-192.png",
        "/static/icons/icon-512.png",
      ])
    )
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.pathname.startsWith("/api/")) return;
  const isAppShell =
    url.pathname === "/" ||
    url.pathname === "/static/app.js" ||
    url.pathname === "/static/styles.css";
  if (isAppShell) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open("organoid-agent-v24").then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }
  event.respondWith(
    caches.match(request).then(
      (cached) =>
        cached ||
        fetch(request).then((response) => {
          const copy = response.clone();
          caches.open("organoid-agent-v24").then((cache) => cache.put(request, copy));
          return response;
        })
    )
  );
});
