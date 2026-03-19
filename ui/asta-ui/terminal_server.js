import { WebSocketServer } from 'ws';
import { spawn } from 'child_process';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Priority: Argument > Env > PORTABLE_EXECUTABLE_DIR > Default (Dev relative)
const ROOT_DIR = process.argv[2] 
  || process.env.ROOT_DIR 
  || process.env.PORTABLE_EXECUTABLE_DIR 
  || path.resolve(__dirname, '../../');
const BACKEND_DIR = ROOT_DIR;

const wss = new WebSocketServer({ port: 8001 });
console.log(`[Terminal Server] Root set to: ${ROOT_DIR}`);

let backendProc = null;

let lastTotalTime = 0;
let lastIdleTime = 0;

function getCpuUsage() {
    const cpus = os.cpus();
    let totalTime = 0;
    let idleTime = 0;
    cpus.forEach(cpu => {
        for (let type in cpu.times) totalTime += cpu.times[type];
        idleTime += cpu.times.idle;
    });
    const deltaTotal = totalTime - lastTotalTime;
    const deltaIdle = idleTime - lastIdleTime;
    lastTotalTime = totalTime;
    lastIdleTime = idleTime;
    if (deltaTotal === 0) return "0.0";
    return ((1 - deltaIdle / deltaTotal) * 100).toFixed(1);
}

function getDiskUsage() {
    return new Promise((resolve) => {
        const cmd = spawn('wmic', ['logicaldisk', 'where', 'DeviceID="C:"', 'get', 'FreeSpace,Size', '/format:list'], { shell: true });
        let out = '';
        cmd.stdout.on('data', (d) => out += d.toString());
        cmd.on('close', () => {
            const lines = out.split(/\r?\n/);
            let free = 0, size = 1;
            lines.forEach(l => {
                const parts = l.trim().split('=');
                if (parts[0] === 'FreeSpace') free = parseInt(parts[1]);
                if (parts[0] === 'Size') size = parseInt(parts[1]);
            });
            resolve(((size - free) / size * 100).toFixed(1));
        });
        cmd.on('error', () => resolve("0.0"));
    });
}

wss.on('connection', (ws) => {
    // Info awal ke terminal UI
    ws.send(JSON.stringify({ type: 'output', data: `[Asta] Terminal connected.\n[Asta] Working Directory: ${ROOT_DIR}\n` }));

    const shell = spawn('cmd.exe', [], {
        cwd: ROOT_DIR,
        env: process.env,
        shell: true
    });

    shell.stdout.on('data', (data) => ws.send(JSON.stringify({ type: 'output', data: data.toString() })));
    shell.stderr.on('data', (data) => ws.send(JSON.stringify({ type: 'output', data: data.toString() })));

    const statsInterval = setInterval(async () => {
        if (ws.readyState !== 1) return;
        ws.send(JSON.stringify({ type: 'stats', data: { cpu: getCpuUsage(), ram: (((os.totalmem()-os.freemem())/os.totalmem())*100).toFixed(1), disk: await getDiskUsage() } }));
    }, 2000);

    ws.on('message', (message) => {
        const input = message.toString().trim();
        const cmd = input.toLowerCase();

        if (cmd === 'cls' || cmd === 'clear') {
            ws.send(JSON.stringify({ type: 'clear' }));
            shell.stdin.write('\n');
            return;
        }

        if (cmd === 'start backend') {
            if (backendProc) {
                ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Backend is already running.\n" }));
                return;
            }
            const pythonPath = path.join(BACKEND_DIR, 'venv', 'Scripts', 'python.exe');
            if (!fs.existsSync(pythonPath)) {
                ws.send(JSON.stringify({ type: 'output', data: `[Error] Python not found at: ${pythonPath}\n` }));
                return;
            }
            ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Starting Backend...\n" }));
            
            backendProc = spawn(pythonPath, ['-m', 'uvicorn', 'api:app', '--host', '0.0.0.0', '--port', '8000'], {
                cwd: BACKEND_DIR,
                shell: true
            });

            backendProc.stdout.on('data', (d) => ws.send(JSON.stringify({ type: 'output', data: `[BACKEND] ${d}` })));
            backendProc.stderr.on('data', (d) => ws.send(JSON.stringify({ type: 'output', data: `[BACKEND] ${d}` })));
            backendProc.on('close', () => {
                ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Backend process closed.\n" }));
                backendProc = null;
            });
            return;
        }

        if (cmd === 'stop backend') {
            if (backendProc) {
                spawn("taskkill", ["/pid", backendProc.pid, "/f", "/t"]);
                backendProc = null;
                ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Backend stopped.\n" }));
            } else {
                ws.send(JSON.stringify({ type: 'output', data: "[Terminal] Backend is not running.\n" }));
            }
            return;
        }

        if (cmd === 'help') {
            ws.send(JSON.stringify({ type: 'output', data: "\nCommands:\n  start backend - Menjalankan server python\n  stop backend  - Menghentikan server python\n  cls           - Bersihkan layar\n  dir           - Lihat isi folder\n" }));
            return;
        }

        shell.stdin.write(input + '\n');
    });

    ws.on('close', () => {
        clearInterval(statsInterval);
        shell.kill();
    });
});
