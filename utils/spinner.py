import sys
import threading
import itertools
import time
import random

class Spinner:
    DEFAULT_MESSAGES = [
        "Sedang berpikir... jangan ganggu, lagi buffering...",
        "Mengumpulkan niat... mohon tunggu sebentar...",
        "Asta sedang menyeduh jawaban terbaik...",
        "Asta baru bangun tidur... silahkan tunggu...",
        "Memanggil inspirasi dari model lain...",
        "Sedang merangkai kata biar kelihatan pintar...",
        "Loading... otak Asta lagi pemanasan...",
        "Asta sedang membuat kopi susu...",
        "Asta Sedang ngemil microchip..."
        "Asta sedang membeli susu..."
        "Mencari jawaban di antara 1 dan 0...",
        "Sedang konsultasi dengan sel otak virtual...",
        "Memproses... biar jawabannya nggak asal...",
        "Asta sedang tidur...",
        "Asta sedang malas berpikir... dibujuk dulu.",
        "Asta lagi tidur... sedang mimpi jadi super AI.",
        "Asta sedang makan data... biar jawabannya bergizi.",
        "Asta ngopi dulu... biar responnya melek.",
        "Asta lagi stretching neuron... pemanasan dulu.",
        "Asta sedang mencari ide di kulkas...",
        "Asta lagi bengong menatap void digital...",
        "Asta sedang merakit jawaban dengan hati-hati...",
        "Reboot otak sebentar...",
        "Asta lagi update mood ke mode pintar.",
        "Asta sedang menyisir pikiran yang berantakan…",
        "Asta lagi isi ulang energi kosmik...",
        "Asta sedang mengetuk pintu inspirasi...",
        "Asta lagi debugging eksistensi...",
        "Asta sedang menata kata biar keren.",
        "Asta lagi tarik napas digital...",
        "Asta sedang konsultasi dengan neuron senior...",
        "Asta lagi memanggil ide dari dimensi lain...",
        "Asta sedang menyusun jawaban level premium...",
        "Asta lagi bangun dari tidur siang algoritmik...",
        "Never gonna give you up, Never gonna let you down...",
    ]

    def __init__(self, messages=None, delay=0.1):
        if messages is None:
            self.messages = Spinner.DEFAULT_MESSAGES
        elif isinstance(messages, str):
            self.messages = [messages]
        else:
            self.messages = messages

        self.message_generator = itertools.cycle(self.messages)
        self.spinner_generator = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.delay = delay
        self.running = False
        self.spinner_thread = None
        self.current_message = next(self.message_generator)

    def _spinner_task(self):
        last_message_change_time = time.time()
        while self.running:
            if time.time() - last_message_change_time >= 5:
                self.current_message = random.choice(self.messages)
                last_message_change_time = time.time()
            
            sys.stdout.write('\r' + ' ' * 70 + '\r' + f"{next(self.spinner_generator)} {self.current_message}") # Aggressively clear and then write
            sys.stdout.flush()
            time.sleep(self.delay)
        sys.stdout.write('\r' + ' ' * 70 + '\r')
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spinner_task)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        if self.spinner_thread and self.spinner_thread.is_alive():
            self.spinner_thread.join(timeout=self.delay * 2)
        sys.stdout.write('\r' + ' ' * 70 + '\r')
        sys.stdout.flush()

    def set_message(self, message):
        self.current_message = message
        self.message_generator = itertools.cycle([message])
