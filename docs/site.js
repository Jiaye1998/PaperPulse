const copyButton = document.querySelector("[data-copy]");

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const originalLabel = copyButton.textContent;
    try {
      await navigator.clipboard.writeText(copyButton.dataset.copy || "");
      copyButton.textContent = "Copied";
    } catch {
      copyButton.textContent = "Copy failed";
    }
    window.setTimeout(() => {
      copyButton.textContent = originalLabel;
    }, 1800);
  });
}

const year = document.querySelector("#year");
if (year) year.textContent = String(new Date().getFullYear());
