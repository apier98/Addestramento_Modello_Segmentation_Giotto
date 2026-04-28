import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Inferenza YOLO da Webcam per la Tesi")
    
    # NOTA BENE: Il tipo è 'int' (intero), non 'str' (stringa) come per il video.
    # 0 è quasi sempre la webcam integrata del portatile.
    parser.add_argument("--webcam", type=int, default=0, help="ID della webcam (es. 0, 1, 2)")
    parser.add_argument("--model", type=str, default="runs/yolo26n-seg-run/weights/best.pt", help="Percorso del modello .pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza")
    
    args = parser.parse_args()

    # 1. Caricamento modello
    print(f"Caricamento del modello: {args.model}")
    model = YOLO(args.model)

    print(f"Avvio streaming dalla webcam ID: {args.webcam}...")
    print("Premi il tasto 'q' sulla finestra del video per chiudere il programma.")

    # 2. Esecuzione inferenza
    # source = args.webcam (passa il numero 0)
    # show = True (obbligatorio per vedere la finestra in tempo reale)
    # stream = True (obbligatorio per evitare memory leak con flussi infiniti)
    results = model.predict(
        source=args.webcam, 
        conf=args.conf, 
        device="cpu", 
        show=True, 
        stream=True 
    )

    # 3. Mantenimento del flusso
    # A differenza del comando da terminale, in Python DEVI ciclare i risultati,
    # altrimenti lo script elabora il primo frame e si chiude immediatamente.
    for r in results:
        # Se volessi salvare i dati in un CSV in tempo reale, 
        # inseriresti qui la logica di estrazione dell'area vista in precedenza.
        if r.masks is not None:
            num_oggetti = len(r.masks)
            # Stampiamo a terminale per confermare che l'IA sta "guardando"
            print(f"Rilevati {num_oggetti} oggetti in tempo reale.", end="\r")

if __name__ == "__main__":
    main()