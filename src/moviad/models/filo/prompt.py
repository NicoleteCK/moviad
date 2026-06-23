status_normal = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

status_abnormal_winclip = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

anomaly_status_general = ["anomaly", "damage", "broken", "defect", "contamination"]

mvtec_anomaly_detail_gpt = {
    "carpet": "discoloration in a specific area,irregular patch or section with a different texture,frayed edges or unraveling fibers,burn mark or scorching",
    "grid": "crooked,cracks,excessive gaps,discoloration,deformation,missing,inconsistent spacing between grid elements,corrosion,visible signs,chipping",
    "leather": "scratches,discoloration,creases,uneven texture,tears,brittleness,damage,seams,heat damage,mold",
    "tile": "chipped,irregularities,discoloration,efflorescence,warping,missing,depressions,lippage,fungus,damage",
    "wood": "knots,warping,cracks along the grain,mold growth on the surface,staining from water damage,wood rot,woodworm holes,rough patches,protruding knots",
    "bottle": "cracked large,cracked small,dented large,dented small,leaking,discolored,deformed,missing cap,excessive condensation,unusual odor",
    "cable": "twisted,knotted cable strands,detached connectors,excessive stretching,dents,corrosion,scorching along the cable,exposed conductive material",
    "capsule": "irregular shape,discoloration coloring,crinkled,uneven seam,condensation inside the capsule,foreign particles,unusually soft or hard",
    "hazelnut": "fungal growth,unusual discoloration,rotten or foul odor emanating,insect infestation,wetness,misshapen shell,unusually thin,contaminants,unusual texture",
    "metal nut": "cracks,irregular threading,corrosion,missing,distortion,signs of discoloration,excessive wear on contact surfaces,inconsistent texture",
    "pill": "irregular shape,crumbling texture,excessive powder,Uneven coating,presence of air bubbles,disintegration,abnormal specks",
    "screw": "rust on the surface,bent,damaged threads,stripped threads,deformed top,coating damage,uneven grooves,inconsistent size",
    "toothbrush": "loose bristles,uneven bristle distribution,excessive shedding of bristles,staining on the bristles,abrasive texture,irregularities in the shape",
    "transistor": "burn marks,detached leads,signs of corrosion,irregularities in the shape,presence of cracks or fractures,signs of physical trauma,irregularities in the surface texture",
    "zipper": "bent,frayed,misaligned,excessive stiffness,corroded,detaches,loose,warped",
}

visa_anomaly_detail_gpt = {
    "candle": "cracks or fissures in the wax,Wax pooling unevenly around the wick,tunneling,incomplete wax melt pool,irregular or flickering flame,other,extra wax in candle,wax melded out of the candle",
    "capsules": "uneven capsule size,capsule shell appears brittle,excessively soft,dents,condensation,irregular seams or joints,specks",
    "cashew": "uneven coloring,fungal growth,presence of foreign objects,unusual texture,empty shells,signs of moisture,stuck together",
    "chewinggum": "consistency,presence of foreign objects,uneven coloring,excessive hardness,similar colour spot",
    "fryum": "irregular shape,unusual odor,uneven coloring,unusual texture,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1": "uneven shape ,small scratches,small cracks,uneven coloring,signs of insect infestation,uneven texture,Unusual consistency",
    "macaroni2": "irregular shape,small scratches,presence of foreign particles,excessive moisture,Signs of infestation,small cracks,unusual texture",
    "pcb1": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb2": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb3": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb4": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pipe fryum": "uneven shape,presence of foreign objects,different colour spot,unusual odor,empty interior,unusual texture,similar colour spot,stuck together",
}

cps_anomaly_detail_gpt = {
    "ceramic package substrate": "cracks,chipped edges,delamination,voids,scratches,contamination,discoloration,warpage,missing metallization,excess metallization,broken traces,metal bridging,metal peeling,via defects,corrosion"
}

miic_anomaly_detail_gpt = {
    "integrated circuit metal layer": "open circuits,metal bridging,line breaks,line thinning,line widening,missing metal,excess metal,voids,cracks,particle contamination,residue,scratches,overetching,underetching,stringers,misalignment"
}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    mvtec_anomaly_detail_gpt[cls_name] = (
        mvtec_anomaly_detail_gpt[cls_name].split(",")
    )

for cls_name in visa_anomaly_detail_gpt.keys():
    visa_anomaly_detail_gpt[cls_name] = (
        visa_anomaly_detail_gpt[cls_name].split(",")
    )

for cls_name in cps_anomaly_detail_gpt.keys():
    cps_anomaly_detail_gpt[cls_name] = (
        cps_anomaly_detail_gpt[cls_name].split(",")
    )

for cls_name in miic_anomaly_detail_gpt.keys():
    miic_anomaly_detail_gpt[cls_name] = (
        miic_anomaly_detail_gpt[cls_name].split(",")
    )

status_abnormal = {}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in mvtec_anomaly_detail_gpt[cls_name]] 

for cls_name in visa_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in visa_anomaly_detail_gpt[cls_name]]

for cls_name in cps_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in cps_anomaly_detail_gpt[cls_name]]

for cls_name in miic_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in miic_anomaly_detail_gpt[cls_name]]

mvtec_obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

visa_obj_list = [
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]

cps_obj_list = [
    "ceramic package substrate",
]

miic_obj_list = [
    "integrated circuit metal layer",
]

dataset_anomaly_maps = {
    "mvtec": mvtec_anomaly_detail_gpt,
    "visa": visa_anomaly_detail_gpt,
    "cps": cps_anomaly_detail_gpt,
    "miic": miic_anomaly_detail_gpt
}

cls_map = {}
for c_name in visa_obj_list:
    cls_map[c_name] = c_name

for c_name in mvtec_obj_list:
    cls_map[c_name] = c_name

for c_name in cps_obj_list:
    cls_map[c_name] = c_name

for c_name in miic_obj_list:
    cls_map[c_name] = c_name

cls_map["pcb1"] = "pcb"
cls_map["pcb2"] = "pcb"
cls_map["pcb3"] = "pcb"
cls_map["pcb4"] = "pcb"
cls_map["macaroni1"] = "macaroni"
cls_map["macaroni2"] = "macaroni"

positions_list = [
    "top left",
    "top",
    "top right",
    "left",
    "center",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
]
