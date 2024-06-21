def load_pvoc_annotations(path):
    import xml.etree.ElementTree as ET
    import glob
    import os
    data = []
    files = glob.glob(os.path.join(path, "*.xml"))
    for file in files:
        root = ET.parse(file).getroot()
        ele = root.find("filename")
        filename = ele.text if ele != None else ""
        ele = root.find("size").find("width")
        width = int(ele.text) if ele != None else 0
        ele = root.find("size").find("height")
        height = int(ele.text) if ele != None else 0
        for o in root.findall("object"):
            bndbox = o.find("bndbox")
            if bndbox == None:
                continue
            ele = o.find("name")
            if ele == None:
                continue
            obj = dict()
            obj["filename"] = filename
            obj["width"] = width
            obj["height"] = height
            obj["name"] = ele.text
            ele = bndbox.find("xmin")
            obj["xmin"] = int(ele.text) if ele != None else 0
            ele = bndbox.find("ymin")
            obj["ymin"] = int(ele.text) if ele != None else 0
            ele = bndbox.find("xmax")
            obj["xmax"] = int(ele.text) if ele != None else 0
            ele = bndbox.find("ymax")
            obj["ymax"] = int(ele.text) if ele != None else 0
            ele = o.find("pose")
            obj["pose"] = ele.text if ele != None else ""
            ele = o.find("truncated")
            obj["truncated"] = int(ele.text) if ele != None else 0
            ele = o.find("difficult")
            obj["difficult"] = int(ele.text) if ele != None else 0
            ele = o.find("occluded")
            obj["occluded"] = int(ele.text) if ele != None else 0
            data.append(obj)
    return data

