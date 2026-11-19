/* Size same-origin players to their content and start each once it enters view. */
(() => {
  function init() {
    for (const frame of document.querySelectorAll(
      "iframe.cp-animation-frame",
    )) {
      function connect() {
        const main = frame.contentDocument?.querySelector("main");
        if (!main) return;
        const resize = () => {
          frame.style.height = `${Math.ceil(main.getBoundingClientRect().height) + 2}px`;
        };
        resize();
        new ResizeObserver(resize).observe(main);
        const visible = new IntersectionObserver(
          (entries) => {
            if (!entries.some((entry) => entry.intersectionRatio >= 0.35))
              return;
            frame.contentWindow.postMessage(
              { type: "causalpy-animation-start" },
              window.location.origin,
            );
            visible.disconnect();
          },
          { threshold: 0.35 },
        );
        visible.observe(frame);
      }
      frame.addEventListener("load", connect, { once: true });
      if (frame.contentDocument?.readyState === "complete") connect();
    }
  }
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
