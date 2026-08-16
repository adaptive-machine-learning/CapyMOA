// Wires up the "Copy page" button added by
// docs/_templates/components/llm-page-actions.html: fetches the page's
// generated Markdown (from sphinx_llm.txt) and copies it to the clipboard.
(function () {
  "use strict";

  function flash(button, label, iconClass) {
    var labelEl = button.querySelector(".llm-copy-page-button__label");
    var iconEl = button.querySelector("i");
    var originalLabel = labelEl.textContent;
    var originalIconClass = iconEl.className;

    labelEl.textContent = label;
    iconEl.className = iconClass;

    window.setTimeout(function () {
      labelEl.textContent = originalLabel;
      iconEl.className = originalIconClass;
    }, 2000);
  }

  function copyPageAsMarkdown(button) {
    var href = button.getAttribute("data-markdown-href");
    fetch(href)
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Failed to fetch " + href + ": " + response.status);
        }
        return response.text();
      })
      .then(function (markdown) {
        return navigator.clipboard.writeText(markdown);
      })
      .then(function () {
        flash(button, "Copied!", "fa-solid fa-check");
      })
      .catch(function (error) {
        console.error("Could not copy page as Markdown:", error);
        flash(button, "Failed to copy", "fa-solid fa-triangle-exclamation");
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".llm-copy-page-button").forEach(function (button) {
      button.addEventListener("click", function () {
        copyPageAsMarkdown(button);
      });
    });
  });
})();
