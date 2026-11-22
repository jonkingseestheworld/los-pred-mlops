// Hide the loading overlay once Shiny has connected and finished its first
// round of outputs ("shiny:idle"), but never flash it away in under MIN_MS.
(function () {
  var MIN_MS = 1000;        // minimum time the loading page stays up
  var FALLBACK_MS = 30000;  // hide anyway if Shiny never goes idle (e.g. server error)
  var started = Date.now();
  var done = false;

  function hide() {
    if (done) return;
    done = true;
    var el = document.getElementById("app-loading");
    if (!el) return;
    var wait = Math.max(0, MIN_MS - (Date.now() - started));
    setTimeout(function () {
      el.classList.add("app-loading--hidden");
      setTimeout(function () { el.remove(); }, 500);
    }, wait);
  }

  window.addEventListener("DOMContentLoaded", function () {
    $(document).one("shiny:idle", hide);   // Shiny fires its events through jQuery
    setTimeout(hide, FALLBACK_MS);
  });
})();
