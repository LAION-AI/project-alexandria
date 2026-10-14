'use strict';
(() => {
  const fields = JSON.parse(document.getElementById('field-data').textContent);
  const byKey = new Map(fields.map(field => [field.key, field]));
  const buttons = Array.from(document.querySelectorAll('[data-field]'));
  const setField = key => {
    const field = byKey.get(key);
    if (!field) return;
    document.getElementById('field-category').textContent = field.category.toUpperCase();
    document.getElementById('field-title').textContent = field.title;
    document.getElementById('field-key').textContent = field.key;
    document.getElementById('field-description').textContent = field.description;
    document.getElementById('field-query').textContent = field.query;
    for (const button of buttons) {
      const selected = button.dataset.field === key;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    }
  };
  for (const button of buttons) button.addEventListener('click', () => setField(button.dataset.field));
})();
