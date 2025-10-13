import os, numpy as np, glob
from PIL import Image
from torch.utils.data import Dataset



def _to_rgb3_from_lwir(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    if arr.dtype == np.uint16: #gestione esplicita caso 16-bit
        arr = (arr / 256).astype(np.uint8)  
        img = Image.fromarray(arr, mode='L')
    else:
        img = img.convert('L')
    return Image.merge('RGB', (img, img, img))



def _parse_line(line: str):
    parts = line.strip().split() # rimuove spazi bianchi a inizo/fine e splitta in blocchi allo spazio
    if len(parts) == 0: return None # linea vuota ignorata
    if parts[0].startswith(('#','//')): return None # commento ignorato
    if len(parts) == 1:          
        # quindi formato compatto e corretto: setXX/VYYY/Ixxxxx (con o senza estensione, verrà gestita dopo)
        base = parts[0]    # esempio setXX/VYYY/Ixxxxx
        return (base, base)  # useremo base per poi cercare i visible/lwir
    elif len(parts) == 2:
        # gestione anche del caso a 2 colonne nel caso volessimo farlo: vis_rel lwir_rel
        # esempio: setXX/VYYY/visible/Ixxxxx setXX/VYYY/lwir/Ixxxxx
        return (parts[0], parts[1])
    else:
        raise ValueError(f"Formato non valido nella riga: {line}")

    

def _ensure_ext(path_no_ext: str, ext: str):
    # se ha già estensione, lascia; altrimenti aggiungi ext (passato da args)
    _, tail = os.path.split(path_no_ext)
    if '.' in tail:
        return path_no_ext
    return path_no_ext + ext


class KAISTDatasetFromList(Dataset):
    #Legge una lista .txt:
    # - 1 colonna: setXX/VYYY/Ixxxxx[.ext?]  (si costruiscono vis/lwir)
    # - 2 colonne: <rel_vis> <rel_lwir>     (si usano direttamente)
    # Costruisce i path assoluti concatenando a data_root.
    
    def __init__(self, data_root, list_file, mode=['rgb','thermal'],
                 rgb_transform=None, th_transform=None,
                 ext_vis='.jpg', ext_lwir='.jpg'):
        self.data_root = os.path.expanduser(data_root)
        self.mode = mode
        self.rgb_transform = rgb_transform
        self.th_transform = th_transform
        self.ext_vis = ext_vis
        self.ext_lwir = ext_lwir

        pairs = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for raw in f:
                pr = _parse_line(raw)
                if pr is None: continue
                vis_rel, th_rel = pr

                if pr[0] == pr[1]:
                    # formato compatto: setXX/VYYY/Ixxxxx → espandi in vis/lwir
                    base_no_ext = pr[0]
                    vis_rel_full = os.path.join(os.path.dirname(base_no_ext), 'visible',
                                                os.path.basename(base_no_ext))
                    th_rel_full  = os.path.join(os.path.dirname(base_no_ext), 'lwir',
                                                os.path.basename(base_no_ext))
                else:
                    vis_rel_full = vis_rel   # gia' pronti
                    th_rel_full  = th_rel

                # assicura estensione corretta 
                vis_rel_full = _ensure_ext(vis_rel_full, self.ext_vis)  
                th_rel_full  = _ensure_ext(th_rel_full, self.ext_lwir)

                vis_abs = os.path.join(self.data_root, vis_rel_full)
                th_abs  = os.path.join(self.data_root, th_rel_full)

                if not (os.path.exists(vis_abs) and os.path.exists(th_abs)):
                    raise FileNotFoundError(
                        f"File mancante: {vis_abs if not os.path.exists(vis_abs) else th_abs}"
                    )
                pairs.append((vis_abs, th_abs))


        if len(pairs) == 0:
            raise RuntimeError(
                f"Nessuna coppia valida in {list_file} rispetto a {self.data_root}"
            )
        
        # print(f"[KAISTDatasetFromList] Trovate {len(pairs)} coppie valide dal file {list_file}")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vis_p, th_p = self.pairs[idx]
        rgb = Image.open(vis_p).convert('RGB')
        th  = _to_rgb3_from_lwir(Image.open(th_p))

        if self.rgb_transform is not None:
            rgb = self.rgb_transform(np.array(rgb))
        if self.th_transform is not None:
            th  = self.th_transform(np.array(th))
        return rgb, th