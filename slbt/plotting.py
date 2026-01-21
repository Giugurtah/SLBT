# slbt/plotting.py
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt  # solo se usi anche la versione matplotlib

from typing import Tuple

from .base import Node

# oppure: from .plotting import compute_tree_layout, _recurse, _custom_round

DEFAULT_COLORS = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4CAF50",
    "#9966FF", "#D81B60", "#00ACC1", "#8D6E63", "#FF9800"
]

def _compute_offsets_for_position(pos: int, depth: int) -> Tuple[float, float]:
    """
    Replica la logica di _update_plot per un singolo nodo.

    Restituisce (l_c_node, r_c_node) per un nodo dato il suo 'position' e depth.
    """
    var = pos
    l_c = 0.0
    r_c = 0.0
    incr = 1.0 / (2 ** depth)

    while var > 1:
        l_c += (-2 * (var % 2) + 1) * incr
        r_c += (2 * (var % 2) - 1) * incr
        incr *= 2.0
        var //= 2

    return l_c, r_c

def compute_tree_layout(root: Node):
    """
    Calcola depth, l_c, r_c a partire dalla radice dell'albero.
    Restituisce (depth, l_c, r_c).
    """
    if root is None:
        return 0, 0.0, 0.0

    max_depth = 0
    max_lc = 0.0
    max_rc = 0.0
    max_impurity_decrease = 0.0

    # DFS esplicita: (node, depth)
    stack = [(root, 0, )]

    while stack:
        node, d = stack.pop()
        if node is None:
            continue

        # aggiorna profondità massima
        if d > max_depth:
            max_depth = d

        if float(node.impurity_decrease) > max_impurity_decrease:
            max_impurity_decrease = node.impurity_decrease

        # calcola offset per questo nodo
        lc_node, rc_node = _compute_offsets_for_position(node.position, d)

        if lc_node > max_lc:
            max_lc = lc_node
        if rc_node > max_rc:
            max_rc = rc_node

        # aggiungi figli
        if node.left is not None:
            stack.append((node.left, d + 1))
        if node.right is not None:
            stack.append((node.right, d + 1))

    return max_depth, max_lc, max_rc, max_impurity_decrease

def _custom_round(value, decimals=2):
    if value is None:
        return None
    if abs(value) >= 0.01:
        return round(float(value), decimals)
    # per valori molto piccoli, uso comunque qualche cifra
    return round(float(value), decimals)

def _recurse(node, homogeneity, root_impurity, root_N):
    dist = ", ".join(
        f"{label}={_custom_round(prob, 2)}"
        for label, prob in zip(node.labels.tolist(), node.distribution.tolist())
    )

    if node.suggested_pruning is not None:
        suggested = 1
    else:
        suggested = 0

    # ---- caso foglia ----
    if node.value is not None:
        gcr = None
        if node.GCR is not None:
            gcr = ", ".join(
                f"{label}={_custom_round(prob, 2)}"
                for label, prob in zip(node.labels.tolist(), node.GCR)
            )
        return {
            "isLeaf": 1,
            "suggested": suggested,
            "distribution": dist,
            "distArray": node.distribution.tolist(),
            "position": node.position,
            "value": str(node.value),
            "impurity": float(node.impurity),
            "labels": int(node.N),
            "impurity_decrement": float(node.impurity_decrease),
            "tree_partial_impurity_reduction": float(node.tree_partial_impurity_reduction),
            "labArray": node.labels.tolist(),
            "gcr": gcr,
        }

    # ---- LIFT ----
    if node.LIFT_1 is not None and node.LIFT_2 is not None:
        # stratificato: omogeneity None/A => LIFT_1/2 sono liste di array per strato
        if node.strat_labels is not None and homogeneity in ("none", "A"):
            lift1 = "\n".join(
                f"{strat_label}: (" +
                ", ".join(
                    f"{label}={_custom_round(prob, 2)}"
                    for label, prob in zip(node.labels, lift_vals)
                ) +
                ")"
                for strat_label, lift_vals in zip(node.strat_labels, node.LIFT_1)
            )
            lift2 = "\n".join(
                f"{strat_label}: (" +
                ", ".join(
                    f"{label}={_custom_round(prob, 2)}"
                    for label, prob in zip(node.labels, lift_vals)
                ) +
                ")"
                for strat_label, lift_vals in zip(node.strat_labels, node.LIFT_2)
            )
        else:
            # caso non stratificato (B/AB o nessuna strat_labels)
            L1 = node.LIFT_1
            L2 = node.LIFT_2
            if isinstance(L1, np.ndarray):
                L1 = L1.tolist()
            if isinstance(L2, np.ndarray):
                L2 = L2.tolist()
            lift1 = ", ".join(
                f"{label}={_custom_round(prob, 2)}"
                for label, prob in zip(node.labels.tolist(), L1)
            )
            lift2 = ", ".join(
                f"{label}={_custom_round(prob, 2)}"
                for label, prob in zip(node.labels.tolist(), L2)
            )
    else:
        lift1 = lift2 = None

    # ---- threshold ----
    if node.strat_labels is None or homogeneity in ("A", "AB"):
        # soglia "semplice": lista di valori che vanno a sinistra
        treshold = ", ".join(str(x) for x in node.treshold)
    else:
        # soglia per strato: mappo strat_label -> lista valori
        treshold = ", ".join(
            f"{strat_label}:[{', '.join(str(x) for x in values)}]"
            for strat_label, values in zip(node.strat_labels, node.treshold)
        )

    return {
        "isLeaf": 0,
        "suggested": suggested,
        "feature": node.feature,
        "distribution": dist,
        "distArray": node.distribution.tolist(),
        "lift1": lift1,
        "lift2": lift2,
        "treshold": treshold,
        "position": node.position,
        "gpi": float(node.gpi),
        "ppi": float(node.ppi),
        "impurity": float(node.impurity),
        "impurity_decrement": float(node.impurity_decrease),
        "tree_partial_impurity_reduction": float(node.tree_partial_impurity_reduction),
        "labels": int(node.N),
        "labArray": node.labels.tolist(),
        "children": [
            _recurse(node.left, homogeneity, root_impurity, root_N),
            _recurse(node.right, homogeneity, root_impurity, root_N),
        ],
    }

def plot_html(model, output_file="tree_visualization.html", title="Decision Tree Visualization",
              color_palette=None):
    """
    Genera un file HTML interattivo per visualizzare l'albero SLBT.

    Parameters
    ----------
    model : SLBT
        Modello già fittato, deve avere .root e .homogeneity.
    output_file : str
        Percorso del file HTML da creare.
    title : str
        Titolo della pagina HTML.
    color_palette : list[str], optional
        Lista di colori per le classi (default: 9 colori predefiniti).
    """
    if model.root is None:
        raise ValueError("Tree is empty: call fit() before plot_html().")

    if color_palette is None:
        color_palette = DEFAULT_COLORS

    # 1) calcola layout (depth, l_c, r_c)
    depth, l_c, r_c, max_imp_decrease = compute_tree_layout(model.root)

    # 2) costruisci JSON per l'albero e i dati di plotting
    tree_dict = _recurse(model.root, model.homogeneity, 1, 1)
    tree_JSON = json.dumps(tree_dict, indent=4)

    dataPlot = {
        "l_c": l_c,
        "r_c": r_c,
        "depth": depth,
        "max_imp_decrease": max_imp_decrease,
        "colors": color_palette,
        "labels": model.root.labels.tolist(),
    }
    plot_JSON = json.dumps(dataPlot, indent=4)

    # 3) HTML (riuso quasi identico al tuo)
    html_content = f""" 
    <!DOCTYPE html> 
    <html lang="en"> 
    <head> 
        <meta charset="UTF-8"> 
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        <title>Decision Tree Visualization</title> 
        <style> 
            body {{ font-family: Arial, sans-serif; text-align: center}}
            .tree-container {{ display: flex; justify-content: center; margin-top: 20px}}
            .node {{ border: 2px solid blue; border-radius: 100%; background-color: lightcyan; color: lightcyan ;display: inline-block;  position: absolute; width: 24px; height: 24px}} 
            .square {{ position: absolute; transform: translate(14px, 14px); z-index: -1}}
            .leaf {{ display: inline-block; font-weight: bolder; position: absolute; line-height: 24px}}
            .d_r:after {{ content: ""; width: 100%; height: 100%; position: absolute; top: 0; left: 0; background: linear-gradient(to top left, transparent calc(50% - 1px), blue, transparent calc(50% + 1px))}}
            .d_l:after {{ content: ""; width: 100%; height: 100%; position: absolute; top: 0; left: 0; background: linear-gradient(to top right, transparent calc(50% - 1px), blue, transparent calc(50% + 1px))}}
            .tooltip {{ position: absolute; z-index: 1; background-color: rgba(0, 0, 0, 0.8); color: white; padding: 8px; border-radius: 5px; white-space: nowrap; visibility: hidden; opacity: 0; transition: opacity 0.3s; font-size: 14px; width: fit-content}}
            .t_l {{ transform: translate(35px, -5px)}}
            .t_r {{ transform: translateX(-100%) translate(-8px, -5px)}}
            .leaf_value{{ display: inline-block; font-weight: bolder; position: absolute; line-height: 24px; transform: translate(6px, 30px)}}
            #legend {{ display: flex; flex-wrap: wrap; gap: 8px; position: absolute; bottom: 3%; width: 100%; justify-content: center; font-family: sans-serif}}
            .legend-item {{ display: flex; align-items: center; gap: 6px; margin-right: 12px }}
            .color-box {{ width: 16px; height: 16px; border: 1px solid #000; box-sizing: border-box}}
            .impurity-line {{ position: absolute; width: 90%; border-top: 1px dashed #cccccc; z-index: -2; transition: border-color 0.3s, z-index 0s; transform: translateY(12px);}}
            .suggested-line {{ position: absolute; width: 90%; border-top: 1px dashed #e61414; z-index: -2; transition: border-color 0.3s, z-index 0s; transform: translateY(12px);}}
            .impurity-line.highlighted {{  border-top: 2px dashed #4a90e2; z-index: -1;}}
            .suggested-line.highlighted {{  border-top: 2px dashed #e61414; z-index: -1;}}
            .impurity-label-left {{  position: absolute; font-size: 11px; color: #666; background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; background-color: 0.3s; z-index: 0s; font-weight: 0.3s; right: 0%; transform: translateY(12px);}}
            .impurity-label-right {{position: absolute; font-size: 11px; color: #666; background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; background-color: 0.3s; z-index: 0s; font-weight: 0.3s; left: 0%; transform: translateY(12px);}}
            .suggested-label-left {{  position: absolute; font-size: 11px; color: #e61414; background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; background-color: 0.3s; z-index: 1; font-weight: 0.3s; right: 0%; transform: translateY(12px);}}
            .suggested-label-right {{ position: absolute; font-size: 11px; color: #e61414; background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; border-radius: 3px; transition: color 0.3s; background-color: 0.3s; z-index: 1; font-weight: 0.3s; left: 0%; transform: translateY(12px);}}
            .impurity-label-left.highlighted {{ color: #ffffff; background-color: rgba(74, 144, 226, 0.95); font-weight: bold; z-index: 11;}}
            .impurity-label-right.highlighted {{ color: #ffffff; background-color: rgba(74, 144, 226, 0.95); font-weight: bold; z-index: 11;}}
            .suggested-label-left.highlighted {{color: #ffffff; background-color: #e61414; font-weight: bold; z-index: 11;}}
            .suggested-label-right.highlighted {{ color: #ffffff; background-color: #e61414; font-weight: bold; z-index: 11;}}
            .square.highlighted-branch:after {{background: linear-gradient(to top left, transparent calc(50% - 1px), orange, transparent calc(50% + 1px)) !important;}}
            .d_l.highlighted-branch:after {{background: linear-gradient(to top right, transparent calc(50% - 1px), orange, transparent calc(50% + 1px)) !important;}}
        </style>
    </head> 
    <body> 
        <h1>{title}</h1> 
        <div class="tree-container" id="tree"></div> 

        <div id="tooltip" class="tooltip"></div>
        
        <script> 
            const treeData = {tree_JSON}; 
            const plotData = {plot_JSON};
            
            function iter(node, left, d, h, prev_impurity) {{
                let impurityLineElement = document.createElement("div");
                impurityLineElement.style.top = (15 + node.impurity_decrement * h) + "%";
                impurityLineElement.setAttribute("data-node-id", node.position); 
                
                let impurityLabelLeft = document.createElement("div");
                impurityLabelLeft.style.top = (15 + node.impurity_decrement * h - 1) + "%";
                impurityLabelLeft.innerText = "V_t(T): " + node.impurity_decrement.toFixed(3);
                impurityLabelLeft.setAttribute("data-node-id", node.position); 

                let impurityLabelRight = document.createElement("div");
                impurityLabelRight.style.top = (15 + node.impurity_decrement * h - 1) + "%";
                impurityLabelRight.innerText = "V(T): " + node.tree_partial_impurity_reduction.toFixed(3);
                impurityLabelRight.setAttribute("data-node-id", node.position); 

                if(node.suggested == 1) {{
                    impurityLineElement.classList.add("suggested-line");
                    impurityLabelLeft.classList.add("suggested-label-left");
                    impurityLabelRight.classList.add("suggested-label-right");
                }} else {{
                    impurityLineElement.classList.add("impurity-line");
                    impurityLabelLeft.classList.add("impurity-label-left");
                    impurityLabelRight.classList.add("impurity-label-right");
                }}

                tree.appendChild(impurityLabelLeft);
                tree.appendChild(impurityLabelRight);
                tree.appendChild(impurityLineElement);

                if(node.impurity_decrement != 0.0){{
                    if(node.position % 2 === 0){{
                        let rbranchElement = document.createElement("div")
                        rbranchElement.classList.add("square")
                        rbranchElement.classList.add("d_r")
                        rbranchElement.style.left = left + "%"
                        rbranchElement.style.top = 15 + prev_impurity*h + "%"
                        rbranchElement.style.height = (node.impurity_decrement-prev_impurity)*h + "%"
                        rbranchElement.style.width = d + "%"
                        rbranchElement.setAttribute("data-node-id", node.position);  
                        rbranchElement.setAttribute("data-impurity-decrease", node.impurity_decrement);  


                        tree.appendChild(rbranchElement)
                    }} else {{
                        let lbranchElement = document.createElement("div")
                        lbranchElement.classList.add("square")
                        lbranchElement.classList.add("d_l")
                        lbranchElement.style.right = (100-left) + "%"
                        lbranchElement.style.top = 15 + prev_impurity*h + "%"
                        lbranchElement.style.height = (node.impurity_decrement-prev_impurity)*h + "%"
                        lbranchElement.style.width = d + "%"
                        lbranchElement.setAttribute("data-node-id", node.position);  
                        lbranchElement.setAttribute("data-impurity-decrease", node.impurity_decrement); 

                        tree.appendChild(lbranchElement) 
                    }}
                }}
                if(node.isLeaf == 1){{
                    let nodeElement = document.createElement("div")
                    nodeElement.classList.add("leaf")
                    nodeElement.style.left = left + "%"
                    nodeElement.style.top = 15 + node.impurity_decrement*h + "%"
                    nodeElement.setAttribute("data-node-id", node.position);  
                    nodeElement.setAttribute("data-impurity-decrease", node.impurity_decrement); 

                    nodeElement.onmouseover = function(event) {{
                        showTooltip(event, node, 15 + node.impurity_decrement*h, left); 
                        highlightImpurityLine(node.position);  
                        highlightSubtree(node.impurity_decrement);  // ← AGGIUNTO
                    }};
                    nodeElement.onmouseout = function() {{
                        hideTooltip();
                        unhighlightImpurityLine(node.position); 
                        unhighlightSubtree();  // ← AGGIUNTO
                    }};
                    
                    let nodeValue = document.createElement("div")
                    nodeValue.classList.add("leaf_value")
                    nodeValue.style.left = left + "%"
                    nodeValue.style.top = 15 + node.impurity_decrement*h + "%"
                    nodeValue.innerText = node.value

                    let canvas = document.createElement("canvas");
                    canvas.width = 36;
                    canvas.height = 36;
                    canvas.style.position = "absolute";
                    canvas.style.top = "-6px";
                    canvas.style.left = "-6px";
                    nodeElement.appendChild(canvas);

                    tree = document.getElementById("tree")
                    tree.appendChild(nodeElement)
                    tree.appendChild(nodeValue)

                    if (node.distribution) {{
                        drawPieChart(canvas, node.distArray, node.labArray);
                    }}
                    return
                }}

                let nodeElement = document.createElement("div")
                nodeElement.classList.add("node")
                nodeElement.style.left = left + "%"
                nodeElement.style.top = 15 + node.impurity_decrement*h + "%"
                
                nodeElement.setAttribute("data-node-id", node.position); 
                nodeElement.setAttribute("data-impurity-decrease", node.impurity_decrement);  // ← AGGIUNTO

                nodeElement.onmouseover = function(event) {{
                    showTooltip(event, node, 15 + node.impurity_decrement*h, left); 
                    highlightImpurityLine(node.position);  
                    highlightSubtree(node.impurity_decrement);  // ← AGGIUNTO
                }};
                nodeElement.onmouseout = function() {{
                    hideTooltip();
                    unhighlightImpurityLine(node.position);
                    unhighlightSubtree();  // ← AGGIUNTO
                }};

                let canvas = document.createElement("canvas");
                canvas.width = 36;
                canvas.height = 36;
                canvas.style.position = "absolute";
                canvas.style.top = "-6px";
                canvas.style.left = "-6px";
                nodeElement.appendChild(canvas);

                tree = document.getElementById("tree")
                tree.appendChild(nodeElement)

                if (node.distribution) {{
                    drawPieChart(canvas, node.distArray, node.labArray);
                }}

                iter(node.children[0], left-d/2, d/2, h, node.impurity_decrement)
                iter(node.children[1], left+d/2, d/2, h, node.impurity_decrement)
                return
            }}

            // Legend
            const legendContainer = document.createElement('div');
            legendContainer.id = 'legend';

            for (let i = 0; i < plotData.labels.length; i++) {{
                const item = document.createElement('div');
                item.className = 'legend-item';

                const colorBox = document.createElement('div');
                colorBox.className = 'color-box';
                colorBox.style.backgroundColor = plotData.colors[i];

                const label = document.createElement('span');
                label.textContent = plotData.labels[i];

                item.appendChild(colorBox);
                item.appendChild(label);
                legendContainer.appendChild(item);
            }}
            document.body.appendChild(legendContainer);

            function drawPieChart(canvas, distArray, labArray) {{
                let ctx = canvas.getContext("2d");
                let total = distArray.reduce((sum, val) => sum + val, 0);
                let startAngle = 0;

                distArray.forEach((value, index) => {{
                    let sliceAngle = (value / total) * 2 * Math.PI;

                    ctx.beginPath();
                    ctx.moveTo(18, 18);
                    ctx.arc(18, 18, 16, startAngle, startAngle + sliceAngle);
                    ctx.closePath();
                    plotData.labels.forEach((valueLab, indexLab) => {{
                        if(valueLab == labArray[index]) {{
                            ctx.fillStyle = plotData.colors[indexLab % plotData.colors.length];
                        }}
                    }})
                    ctx.fill();

                    startAngle += sliceAngle;
                }});

                ctx.beginPath();
                ctx.arc(18, 18, 17, 0, 2 * Math.PI);
                ctx.strokeStyle = "blue";
                ctx.lineWidth = 2;
                ctx.stroke();
            }}

            function highlightImpurityLine(nodeId) {{
                // If the line is not suggested
                const lines = document.querySelectorAll('.impurity-line[data-node-id="' + nodeId + '"]');
                lines.forEach(line => {{
                    line.classList.add('highlighted');
                }});
                
                const labels_l = document.querySelectorAll('.impurity-label-left[data-node-id="' + nodeId + '"]');
                labels_l.forEach(label => {{
                    label.classList.add('highlighted');
                }});

                const labels_r = document.querySelectorAll('.impurity-label-right[data-node-id="' + nodeId + '"]');
                labels_r.forEach(label => {{
                    label.classList.add('highlighted');
                }});

                // If the line is suggested
                const s_lines = document.querySelectorAll('.suggested-line[data-node-id="' + nodeId + '"]');
                s_lines.forEach(line => {{
                    line.classList.add('highlighted');
                }});
                
                const s_labels_l = document.querySelectorAll('.suggested-label-left[data-node-id="' + nodeId + '"]');
                s_labels_l.forEach(label => {{
                    label.classList.add('highlighted');
                }});

                const s_labels_r = document.querySelectorAll('.suggested-label-right[data-node-id="' + nodeId + '"]');
                s_labels_r.forEach(label => {{
                    label.classList.add('highlighted');
                }});
                
            }}

            function unhighlightImpurityLine(nodeId) {{
                // If the node is not suggested
                const lines = document.querySelectorAll('.impurity-line[data-node-id="' + nodeId + '"]');
                lines.forEach(line => {{
                    line.classList.remove('highlighted');
                }});
                
                const labels_l = document.querySelectorAll('.impurity-label-left[data-node-id="' + nodeId + '"]');
                labels_l.forEach(label => {{
                    label.classList.remove('highlighted');
                }});

                const labels_r = document.querySelectorAll('.impurity-label-right[data-node-id="' + nodeId + '"]');
                labels_r.forEach(label => {{
                    label.classList.remove('highlighted');
                }});

                // If the node is suggested
                const s_lines = document.querySelectorAll('.suggested-line[data-node-id="' + nodeId + '"]');
                s_lines.forEach(line => {{
                    line.classList.remove('highlighted');
                }});
                
                const s_labels_l = document.querySelectorAll('.suggested-label-left[data-node-id="' + nodeId + '"]');
                s_labels_l.forEach(label => {{
                    label.classList.remove('highlighted');
                }});

                const s_labels_r = document.querySelectorAll('.suggested-label-right[data-node-id="' + nodeId + '"]');
                s_labels_r.forEach(label => {{
                    label.classList.remove('highlighted');
                }});
            }}

            function showTooltip(event, node, top, left) {{
                tooltip = document.getElementById('tooltip')
                wh = window.screen.height
                th = tooltip?.clientHeight

                if(node.isLeaf == 1){{
                    tooltip.innerText = "Id: " + node.position + "\\nN: " + node.labels + "\\nClass distribution: [" + node.distribution +  "]\\nGCR: [" + node.gcr + "]\\nImpurty: " + node.impurity.toFixed(3) + "\\nImpurity decrease: " + node.impurity_decrement.toFixed(3) + "\\nPrediction: "+ node.value;
                }} else {{
                    if(node.lift1 && node.lift2) {{
                        tooltip.innerText = "Id: " + node.position + "\\nN: " + node.labels +  "\\nClass distribution: [" + node.distribution + "]\\nLIFT left: [" + node.lift1 + "]\\nLIFT right: [" + node.lift2 + "]\\nFeature: " + node.feature + "\\nThreshold left: [" + node.treshold + "]\\nGpi: " + node.gpi.toFixed(3) + "\\nPpi: " + node.ppi.toFixed(3)  + "\\nImpurty: " + node.impurity.toFixed(3) + "\\nImpurity decrease: " + node.impurity_decrement.toFixed(3);
                    }} else {{
                        tooltip.innerText = "Id: " + node.position + "\\nN: " + node.labels +  "\\nClass distribution: [" + node.distribution + "]\\nFeature: " + node.feature + "\\nThreshold left: [" + node.treshold + "]\\nGpi: " + node.gpi.toFixed(3) + "\\nPpi: " + node.ppi.toFixed(3)  + "\\nImpurty: " + node.impurity.toFixed(3) + "\\nImpurity decrease: " + node.impurity_decrement.toFixed(3);
                    }}
                }}

                if((top/100) + th/wh < 0.90){{
                    if(left > 50){{
                        tooltip.style.left = left + "%"
                        tooltip.style.top = top + "%"
                        tooltip.style.bottom = "auto"
                        tooltip.classList.add("t_r")
                    }} else {{
                        tooltip.style.left = left + "%"
                        tooltip.style.top = top + "%"
                        tooltip.style.bottom = "auto"
                        tooltip.classList.add("t_l")
                    }}
                }} else {{
                    if(left > 50){{
                        tooltip.style.left = left + "%"
                        tooltip.style.bottom = 1 + "%"
                        tooltip.style.top = "auto"
                        tooltip.classList.add("t_r")
                    }} else {{
                        tooltip.style.left = left + "%"
                        tooltip.style.bottom = 1 + "%"
                        tooltip.style.top = "auto"
                        tooltip.classList.add("t_l")
                    }}
                }}

                tooltip.style.visibility = 'visible'
                tooltip.style.opacity = 1
            }}

            function hideTooltip() {{
                tooltip = document.getElementById('tooltip')
                tooltip.classList.remove("t_r")
                tooltip.classList.remove("t_l")
                tooltip.style.visibility = 'hidden'
                tooltip.style.opacity = 0
            }}

            function getDirectChildren(nodeId) {{ // AGGIUNTO
                return [2 * nodeId, 2 * nodeId + 1];
            }}

            function highlightSubtree(currentImpurityDecrease) {{
                const allNodes = document.querySelectorAll('[data-node-id]');
                const nodesToHighlight = new Set();
                
                allNodes.forEach(element => {{
                    const impDec = parseFloat(element.getAttribute('data-impurity-decrease'));
                    const nodeId = parseInt(element.getAttribute('data-node-id'));
                    
                    if (impDec < currentImpurityDecrease) {{
                        nodesToHighlight.add(nodeId);
                    }}
                }});
                
                const nodesToHighlightCopy = new Set(nodesToHighlight);
                nodesToHighlightCopy.forEach(nodeId => {{
                    const children = getDirectChildren(nodeId);
                    children.forEach(childId => {{
                        nodesToHighlight.add(childId);
                    }});
                }});
                
                nodesToHighlight.forEach(nodeId => {{
                    const nodeElements = document.querySelectorAll('.node[data-node-id="' + nodeId + '"]');
                    nodeElements.forEach(
                        el => {{
                            el.classList.add('highlighted-node')

                            // Ridisegna il bordo del canvas in rosso
                            const canvas = el.querySelector('canvas');
                            if (canvas) {{
                                const ctx = canvas.getContext('2d')
                                ctx.beginPath();
                                ctx.arc(18, 18, 17, 0, 2 * Math.PI);
                                ctx.strokeStyle = "orange";
                                ctx.lineWidth = 2;
                                ctx.stroke();
                            }}
                        }}
                    );
                    
                    const leafElements = document.querySelectorAll('.leaf[data-node-id="' + nodeId + '"]');
                    leafElements.forEach(
                        el => {{
                            el.classList.add('highlighted-node')

                            // Ridisegna il bordo del canvas in rosso
                            const canvas = el.querySelector('canvas');
                            if (canvas) {{
                                const ctx = canvas.getContext('2d')
                                ctx.beginPath();
                                ctx.arc(18, 18, 17, 0, 2 * Math.PI);
                                ctx.strokeStyle = "orange";
                                ctx.lineWidth = 2;
                                ctx.stroke();
                            }}
                        }}
                    );
                    
                    const branchElements = document.querySelectorAll('.square[data-node-id="' + nodeId + '"]');
                    branchElements.forEach(el => el.classList.add('highlighted-branch'));
                }});
            }}

            function unhighlightSubtree() {{
                const highlightedNodes = document.querySelectorAll('.highlighted-node');
                highlightedNodes.forEach(
                    el => {{
                        el.classList.remove('highlighted-node')

                        // Ridisegna il bordo del canvas in rosso
                            const canvas = el.querySelector('canvas');
                            if (canvas) {{
                                const ctx = canvas.getContext('2d')
                                ctx.beginPath();
                                ctx.arc(18, 18, 17, 0, 2 * Math.PI);
                                ctx.strokeStyle = "blue";
                                ctx.lineWidth = 2;
                                ctx.stroke();
                            }}
                        }}
                    );
                    
                const highlightedBranches = document.querySelectorAll('.highlighted-branch');
                highlightedBranches.forEach(el => el.classList.remove('highlighted-branch'));
            }}


            d = 80/(plotData.l_c+plotData.r_c);
            h = 65/(plotData.max_imp_decrease);

            iter(treeData, 10+d*plotData.l_c, d, h, 0)
        </script> 
    </body>
    </html> 
    """

    output_path = Path(output_file)
    output_path.write_text(html_content, encoding="utf-8")