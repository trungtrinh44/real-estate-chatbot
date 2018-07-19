export function getSelectionText() {
  const selObj = window.getSelection();
  try {
    const selRange = selObj.getRangeAt(0);
    return selRange;
  } catch (err) {
    return null;
  }
}

export function trimNewLine(x) {
  return x.replace(/^[\r\n]+|[\r\n]+$/gm, '');
}

export function generateRandomString(N) {
  const s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  return Array(N).join().split(',').map(() => s.charAt(Math.floor(Math.random() * s.length)))
    .join('');
}

export function checkParentRelation(parentNode, childNode) {
  if ('contains' in parentNode) {
    return parentNode.contains(childNode);
  }
  return parentNode.compareDocumentPosition(childNode) % 16;
}
export function download(data, filename, type) {
  const file = new Blob([data], { type });
  if (window.navigator.msSaveOrOpenBlob) { // IE10+
    window.navigator.msSaveOrOpenBlob(file, filename);
  } else { // Others
    const a = document.createElement('a');
    const url = URL.createObjectURL(file);
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }, 0);
  }
}
