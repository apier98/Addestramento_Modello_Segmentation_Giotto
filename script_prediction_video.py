import argparse
from ultralytics import YOLO

def main():
    # 1. Configurazione degli argomenti da terminale
    parser = argparse.ArgumentParser(description="Inferenza YOLO per la Tesi - Segmentazione Video")
    parser.add_argument("--video", type=str, required=True, help="Percorso del video di input")
    parser.add_argument("--model", type=str, default="runs/yolo26n-seg-run/weights/best.pt", help="Percorso del modello addestrato")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza minima per considerare una rilevazione valida")
    
    args = parser.parse_args()

    # 2. Caricamento del modello
    print(f"Caricamento del modello da: {args.model}")
    model = YOLO(args.model) #istanza del modello caricando i pesi risultati dall'addestramento come file .pt dentro args.model

    # 3. Esecuzione dell'inferenza
    results = model.predict(source=args.video, conf=args.conf, device= 'cpu', show=True, save=True, stream=True)
    # Nota: stream=True è FONDAMENTALE per i video. Permette di processare un frame alla volta come generatore,        # evitando di saturare la RAM del computer caricando tutto il video in memoria.        # 4. Analisi dei risultati frame per frame
        
    frame_count = 0
    for r in results:
        frame_count += 1 # Contatore dei frame processati
        
        # --- QUI INIZIA IL VERO VALORE DELLO SCRIPT PYTHON ---
        # Se ci sono maschere rilevate nel frame attuale
        if r.masks is not None:     # Scrittura di sicurezza per evitare errori in caso non ci siano maschere rilevate
            numero_oggetti = len(r.masks)   # Conta quante maschere sono state rilevate in questo frame
            print(f"Frame {frame_count}: Trovati {numero_oggetti} oggetti segmentati.") #Stampa a video il numero di oggetti segmentati rilevati in questo frame
            
                # Esempio: puoi estrarre i punti del poligono della prima maschera rilevata
                # poligoni = r.masks.xy
            
            # Se vuoi accedere ai Bounding Box:
            # r.boxes (contiene coordinate, confidenza, classe)

if __name__ == "__main__":
    main()