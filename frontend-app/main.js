const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec } = require('child_process');

// To integrate Rust natively, we would load the compiled node module here:
// const rustAddon = require('./native-rust-logic'); 

function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        titleBarStyle: 'hiddenInset', // Apple native window feel
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    mainWindow.loadFile('index.html');
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});

// Run the Python RL simulation script
ipcMain.handle('run-python-simulation', async () => {
    return new Promise((resolve) => {
        // Execute the python script in the parent directory
        exec('python3 ../train_benchmarks.py', (error, stdout, stderr) => {
            if (error) {
                resolve({ status: "error", error: error.message, stderr: stderr });
                return;
            }
            resolve({ status: "success", output: stdout });
        });
    });
});

// Mode Switching (Comfort, Economy, Performance)
ipcMain.handle('set-mode', async (event, mode) => {
    console.log(`System mode switched via Vortex to: ${mode}`);
    return `Mode ${mode} engaged.`;
});
