import argparse
import csv
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Analisi Componenti con Tracking - Tesi")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="runs/yolo26n-seg-run/weights/best.pt")
    parser.add_argument("--out_csv", type=str, default="analisi_componenti_raggruppati.csv")
    args = parser.parse_args()

    model = YOLO(args.model)
    MARGINE_BORDO = 50 

    with open(args.out_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Aggiungiamo 'Track_ID' per raggruppare in Excel
        writer.writerow(['Frame', 'Track_ID', 'Classe', 'Area_Pixel', 'Stato'])

        # Usiamo model.track per mantenere l'identità degli oggetti
        # persist=True mantiene gli ID tra un frame e l'altro
        results = model.track(source=args.video, conf=0.25, device="cpu", stream=True, show=True, persist=True)
        
        print(f"Inizio tracciamento video: {args.video}")
        
        for frame_idx, r in enumerate(results):
            frame_height, frame_width = r.orig_shape

            # Verifichiamo che ci siano box tracciati (fondamentale per avere l'ID)
            if r.boxes is not None and r.boxes.id is not None:
                masks_data = r.masks.data
                boxes_data = r.boxes.xyxy
                track_ids = r.boxes.id.int().tolist() # Qui prendiamo l'ID univoco del pezzo
                classes = r.boxes.cls
                
                for i in range(len(masks_data)):
                    x1, y1, x2, y2 = boxes_data[i].tolist()
                    t_id = track_ids[i]
                    
                    # Logica del Margine
                    if (x1 <= MARGINE_BORDO or y1 <= MARGINE_BORDO or 
                        x2 >= (frame_width - MARGINE_BORDO) or 
                        y2 >= (frame_height - MARGINE_BORDO)):
                        # Se tocca il bordo, non scriviamo nulla (o scriviamo 'BORDO')
                        continue 
                    
                    area_pixel = masks_data[i].sum().item()
                    nome_classe = model.names[int(classes[i])]
                    
                    # Scriviamo i dati: Excel potrà raggruppare per 'Track_ID'
                    writer.writerow([frame_idx, t_id, nome_classe, area_pixel, "CENTRALE"])
            
            if frame_idx % 50 == 0:
                print(f"Processati {frame_idx} frame...")

    print(f"Analisi completata. File salvato: {args.out_csv}")

if __name__ == "__main__":
    main()