const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    runPythonSimulation: () => ipcRenderer.invoke('run-python-simulation'),
    setMode: (mode) => ipcRenderer.invoke('set-mode', mode)
});
