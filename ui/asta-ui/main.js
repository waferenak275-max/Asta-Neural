import { app, BrowserWindow, Menu, nativeTheme, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { fork } from 'child_process';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isDev = !app.isPackaged;

let mainWindow;
let terminalServer;
let backendProcess;

const ROOT_DIR = isDev 
  ? path.resolve(__dirname, '../../') 
  : (process.env.PORTABLE_EXECUTABLE_DIR || path.dirname(app.getPath('exe')));

const THEMES = {
  light: { bg: '#f7f6f4', text: '#1a1816' },
  dark:  { bg: '#141210', text: '#e8e0d5' }
};

function updateTitleBar(isDark) {
  if (!mainWindow) return;
  const theme = isDark ? THEMES.dark : THEMES.light;
  mainWindow.setTitleBarOverlay({
    color: theme.bg,
    symbolColor: theme.text,
    height: 35
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "Asta Neural",
    icon: path.join(__dirname, 'build', 'icon.ico'),
    titleBarStyle: 'hidden',
    titleBarOverlay: {
        color: nativeTheme.shouldUseDarkColors ? THEMES.dark.bg : THEMES.light.bg,
        symbolColor: nativeTheme.shouldUseDarkColors ? THEMES.dark.text : THEMES.light.text,
        height: 35
    },
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  Menu.setApplicationMenu(null);

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC listener untuk perubahan tema dari UI
ipcMain.on('theme-changed', (event, mode) => {
  updateTitleBar(mode === 'dark');
});

function startTerminalServer() {
  terminalServer = fork(path.join(__dirname, 'terminal_server.js'), [ROOT_DIR], {
    env: { ...process.env, ROOT_DIR }
  });
}

function startBackend() {
  const venvPath = path.join(ROOT_DIR, 'venv', 'Scripts', 'activate.bat');
  if (!fs.existsSync(venvPath)) return;

  const runCmd = `"${venvPath}" && uvicorn api:app --host 0.0.0.0 --port 8000`;
  backendProcess = spawn('cmd.exe', ['/c', runCmd], {
    cwd: ROOT_DIR,
    shell: true,
    detached: false // Memastikan dia terikat ke parent
  });
}

function killProcesses() {
    if (backendProcess) {
        spawn("taskkill", ["/pid", backendProcess.pid, "/f", "/t"]);
    }
    if (terminalServer) {
        terminalServer.kill();
    }
}

app.whenReady().then(() => {
  startTerminalServer();
  startBackend();
  createWindow();

  nativeTheme.on('updated', () => {
    updateTitleBar(nativeTheme.shouldUseDarkColors);
  });
});

app.on('before-quit', () => {
    killProcesses();
});

app.on('window-all-closed', () => {
  killProcesses();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
