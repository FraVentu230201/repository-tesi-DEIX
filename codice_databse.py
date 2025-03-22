import os
import csv
import numpy as np
import scipy.io
import random

import random

def convert_mat_to_csv(mat_path, images_folder, output_csv, val_split=0.2):
    """
    Converte il file .mat di MPII in un CSV e assegna un certo numero di immagini alla validazione.
    
    - `val_split`: percentuale di immagini da usare per la validazione (es. 0.2 = 20%).
    """
    import scipy.io
    import csv

    data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    release = data['RELEASE']
    
    annolist = release.annolist
    train_set = release.img_train

    with open(output_csv, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["img_name", "is_train"]
        for j in range(16):
            header.append(f"x{j}")
            header.append(f"y{j}")
        writer.writerow(header)

        num_samples = len(annolist)
        indices = list(range(num_samples))
        random.shuffle(indices)  # Mescoliamo le immagini

        val_size = int(num_samples * val_split)  # Numero di immagini per il validation set
        val_indices = set(indices[:val_size])  # Selezioniamo i primi `val_size` come validation

        for i in range(num_samples):
            img_info = annolist[i]
            img_name = img_info.image.name if hasattr(img_info.image, 'name') else None
            if not img_name:
                continue

            img_path = os.path.join(images_folder, img_name)
            if not os.path.exists(img_path):
                continue

            is_train = 0 if i in val_indices else 1  # Se è nei val_indices → validation, altrimenti training

            annorect = img_info.annorect
            if annorect is None:
                continue

            if isinstance(annorect, np.ndarray):
                rects = annorect
            else:
                rects = [annorect]

            if len(rects) == 0:
                continue

            person = rects[0]
            if not hasattr(person, 'annopoints'):
                continue

            annopoints = person.annopoints
            if annopoints is None:
                continue

            if isinstance(annopoints, np.ndarray):
                if len(annopoints) == 0:
                    continue
                annopoints = annopoints[0]

            if not hasattr(annopoints, 'point'):
                continue

            points = annopoints.point
            if not isinstance(points, (list, np.ndarray)):
                points = [points]

            coords = np.full((16, 2), -1, dtype=float)

            for p in points:
                j_id = p.id
                if j_id < 0 or j_id > 15:
                    continue
                x_ = float(p.x)
                y_ = float(p.y)
                coords[j_id, 0] = x_
                coords[j_id, 1] = y_

            row = [img_name, is_train]
            for j in range(16):
                row.append(coords[j,0])
                row.append(coords[j,1])

            writer.writerow(row)

    print(f"CSV '{output_csv}' generato con {num_samples - val_size} train e {val_size} validation.")



def main():
    """
    Esempio di uso in un'unica esecuzione:
    1) Imposta i path (file .mat, cartella immagini, CSV di output).
    2) Esegue la conversione.
    """
    # Adatta questi path in base alla tua struttura di cartelle:
    mat_file = "MPII_annotation/mpii_human_pose_v1_u12_1.mat"     # path al tuo .mat
    images_folder = "images"                      # cartella con le immagini .jpg
    output_csv = "mpii_annotations.csv"           # CSV da creare

    convert_mat_to_csv(mat_file, images_folder, output_csv)


if __name__=="__main__":
    main()

convert_mat_to_csv("MPII_annotation/mpii_human_pose_v1_u12_1.mat","images","mpii_annotations.csv")