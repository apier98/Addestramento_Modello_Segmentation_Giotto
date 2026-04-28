import argparse
import csv
import time
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Webcam Logger con filtro bordi")
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--model", type=str, default="runs/yolo26n-seg-run/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default="log_produzione.csv")
    args = parser.parse_args()

    model = YOLO(args.model)
    device = "cpu" # Manteniamo CPU per stabilità come deciso in precedenza
    
    # Definiamo un margine (in pixel) dai bordi dell'inquadratura
    MARGINE_BORDO = 50 # Se un oggetto tocca o supera questo margine, lo consideriamo "parzialmente fuori" e lo ignoriamo

    # Prepariamo il file CSV
    with open(args.out_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Classe', 'Area_Pixel', 'Confidenza'])

        print(f"Avvio stream webcam {args.webcam}... Dati salvati in {args.out_csv}")
        
        # Esecuzione inferenza stream
        results = model.predict(source=args.webcam, conf=args.conf, device=device, show=True, stream=True)

        for r in results:
            # Estraiamo le dimensioni originali del frame (Altezza, Larghezza)
            frame_height, frame_width = r.orig_shape

            if r.masks is not None and r.boxes is not None:
                masks_data = r.masks.data
                boxes_data = r.boxes.xyxy # Coordinate dei Bounding Box [x_min, y_min, x_max, y_max]
                classes = r.boxes.cls
                confs = r.boxes.conf
                
                for i in range(len(masks_data)):
                    # 1. Estrazione coordinate scatola di delimitazione
                    x1, y1, x2, y2 = boxes_data[i].tolist()
                    
                    # 2. LOGICA DI CONTROLLO BORDI
                    # Se l'oggetto tocca o supera la zona di margine, lo ignoriamo
                    if (x1 <= MARGINE_BORDO or 
                        y1 <= MARGINE_BORDO or 
                        x2 >= (frame_width - MARGINE_BORDO) or 
                        y2 >= (frame_height - MARGINE_BORDO)):
                        
                        # Il pezzo è parzialmente fuori inquadratura: saltiamo la registrazione
                        continue 
                    
                    # 3. Se passa il controllo, estraiamo le informazioni
                    area_pixel = masks_data[i].sum().item()
                    nome_classe = model.names[int(classes[i])]
                    confidenza = confs[i].item()
                    timestamp_corrente = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 4. Scrittura su disco
                    writer.writerow([timestamp_corrente, nome_classe, area_pixel, round(confidenza, 2)])
                    
                    # Stampa a terminale per feedback visivo
                    print(f"[{timestamp_corrente}] Registrato pezzo valido: {nome_classe} | Area: {area_pixel}")

if __name__ == "__main__":
    main()