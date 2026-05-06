import argparse
import cv2
from ultralytics import YOLO
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    """
    Analizza un video, mostra l'area in tempo reale, salva opzionalmente
    il video annotato e alla fine genera un CSV e un grafico
    dell'andamento delle aree nel tempo.
    """
    parser = argparse.ArgumentParser(description="Mostra l'area dei componenti in tempo reale, salva il video e genera un grafico finale.")
    parser.add_argument("--video", type=str, required=True, help="Percorso del video da analizzare.")
    parser.add_argument("--model", type=str, default="runs/yolo26n-seg-run/weights/best.pt", help="Percorso del modello YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza per la rilevazione.")
    parser.add_argument("--save_video", type=str, default=None, help="Percorso opzionale dove salvare il video con le annotazioni. Es: output.mp4")
    parser.add_argument("--device", type=str, default="cpu", help="Device da usare per l'inferenza (es. 'cpu', '0'). Ignorato se si usa --use_directml.")
    parser.add_argument("--use_directml", action="store_true", help="Usa il modello ONNX con DirectML per l'accelerazione GPU.")
    parser.add_argument("--out_csv", type=str, default="andamento_aree_live.csv", help="File CSV di output generato.")
    parser.add_argument("--out_plot", type=str, default="andamento_aree_live.png", help="Immagine PNG del grafico generato.")
    args = parser.parse_args()

    # 1. Caricamento del modello
    if args.use_directml:
        if args.model.endswith('.pt'):
            args.model = args.model.replace('.pt', '.onnx')
        print(f"Modalità DirectML attivata. Verrà utilizzato il modello ONNX: {args.model}")
        model = YOLO(args.model, task='segment')
    else:
        print(f"Caricamento del modello da: {args.model}")
        model = YOLO(args.model)

    # 2. Inizializzazione VideoCapture per ottenere le proprietà del video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video sorgente: {args.video}")
        return
    
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 3. Inizializzazione VideoWriter se richiesto
    writer = None
    if args.save_video:
        output_dir = os.path.dirname(args.save_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, video_fps, (video_w, video_h))
        print(f"Il video con le annotazioni verrà salvato in: {args.save_video}")

    # 4. Strutture dati per CSV e grafico
    dati_plot = defaultdict(lambda: {'frames': [], 'areas': []})
    raw_csv_data = [] # Lista per memorizzare i dati grezzi prima di scrivere il CSV

    # 5. Esecuzione del tracking
    print("Avvio analisi video... Premere 'q' sulla finestra per terminare.")
    track_kwargs = {'source': args.video, 'conf': args.conf, 'stream': True, 'persist': True, 'show': False}
    if not args.use_directml:
        track_kwargs['device'] = args.device
    
    results = model.track(**track_kwargs)

    for frame_idx, r in enumerate(results):
        annotated_frame = r.plot()

        if r.boxes is not None and r.boxes.id is not None and r.masks is not None:
            masks_data = r.masks.data
            boxes_coords = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().tolist()

            for i in range(len(masks_data)):
                t_id = track_ids[i]
                area_pixel = int(masks_data[i].sum().item())
                
                # Salvataggio dati per il grafico
                dati_plot[t_id]['frames'].append(frame_idx)
                dati_plot[t_id]['areas'].append(area_pixel)
                
                # Salvataggio dati grezzi per il CSV
                raw_csv_data.append({
                    'Frame': frame_idx,
                    'Track_ID': t_id,
                    'Area_Pixel': area_pixel
                })
                
                # Annotazione live sul frame
                x1, y1, _, y2 = boxes_coords[i]
                text = f"Area: {area_pixel} px"
                text_pos = (int(x1), int(y2) + 20)
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (text_pos[0], text_pos[1] - text_height - 5), (text_pos[0] + text_width, text_pos[1] + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, text, (text_pos[0], text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Analisi Area in Tempo Reale", annotated_frame)
        if writer:
            writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Rilascio risorse video
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\nAnalisi video terminata.")

    # 7. Calcolo medie e generazione CSV
    component_avg_areas = {}
    for t_id, data in dati_plot.items():
        if data['areas']:
            component_avg_areas[t_id] = sum(data['areas']) / len(data['areas'])
        else:
            component_avg_areas[t_id] = 0 # Caso limite, non dovrebbe accadere con dati validi

    print(f"Scrittura dati testuali in: {args.out_csv}")
    with open(args.out_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header aggiornato con la nuova colonna
        csv_writer.writerow(['Frame', 'Track_ID', 'Area_Pixel', 'Area_Media_Componente'])

        for row_data in raw_csv_data:
            t_id = row_data['Track_ID']
            avg_area = component_avg_areas.get(t_id, 0)
            csv_writer.writerow([
                row_data['Frame'],
                row_data['Track_ID'],
                row_data['Area_Pixel'],
                round(avg_area, 2) # Arrotondiamo per leggibilità
            ])
    print(f"Dati testuali salvati in: {args.out_csv}")

    # 8. Generazione grafico
    print("Generazione del grafico in corso...")

    plt.figure(figsize=(14, 8))
    for t_id, data in dati_plot.items():
        plt.plot(data['frames'], data['areas'], marker='o', markersize=4, linestyle='-', label=f'Track ID {t_id}')

    plt.xlabel('Numero Frame (Tempo)', fontsize=12)
    plt.ylabel('Area Segmentata (Pixel)', fontsize=12)
    plt.title("Andamento dell'Area dei Componenti Tracciati nel Tempo", fontsize=14, fontweight='bold')
    plt.legend(title='Componenti', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.out_plot)
    print(f"Grafico salvato con successo in: {args.out_plot}")
    plt.show()

if __name__ == "__main__":
    main()