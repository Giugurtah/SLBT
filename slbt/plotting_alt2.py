# slbt/plotting.py

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

def plot_html(
    model,
    output_file: str = "tree.html",
    title: str = "SLBT Decision Tree",
    visual_pruning: bool = False
):
    """
    Generate interactive HTML visualization of SLBT tree.
    
    Parameters
    ----------
    model : SLBT
        Trained SLBT model
    output_file : str
        Output HTML file path
    title : str
        Title for the visualization
    visual_pruning : bool
        If True, use visual pruning layout with vertical positioning
        based on impurity decrease
    """
    if model.root is None:
        raise ValueError("Model has not been fitted yet")
    
    # Build tree structure - pass root_n explicitly
    tree_data = _build_tree_structure(model.root, visual_pruning=visual_pruning, root_n=model.root.N)
    
    # Calculate layout
    if visual_pruning:
        layout = _calculate_visual_pruning_layout(tree_data)
    else:
        layout = _calculate_standard_layout(tree_data)
    
    # Generate HTML
    html = _generate_html(tree_data, layout, title, visual_pruning)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def _build_tree_structure(node, parent_id=None, visual_pruning=False, root_n=None):
    """Build tree structure as nested dict."""
    if node is None:
        return None
    
    # Get root_n from first call (root node)
    if root_n is None:
        root_n = node.N
    
    # Calculate proportion
    proportion = node.N / root_n if root_n > 0 else 1.0
    
    node_data = {
        'id': node.position,
        'parent_id': parent_id,
        'is_leaf': node._is_leaf_node(),
        'impurity': node.impurity,
        'proportion': proportion,
        'n_samples': node.N,
        'distribution': node.distribution.tolist() if node.distribution is not None else [],
        'feature': node.feature if not node._is_leaf_node() else None,
        'threshold': list(node.treshold) if (not node._is_leaf_node() and node.treshold is not None) else None,
        'prediction': node.value if node._is_leaf_node() else None,
        'gpi': node.gpi,
        'ppi': node.ppi,
        'LIFT_1': node.LIFT_1,
        'LIFT_2': node.LIFT_2,
        'GCR': node.GCR
    }
    
    # Add children
    children = []
    if not node._is_leaf_node():
        if node.left:
            children.append(_build_tree_structure(node.left, node.position, visual_pruning, root_n))
        if node.right:
            children.append(_build_tree_structure(node.right, node.position, visual_pruning, root_n))
    
    node_data['children'] = children
    
    return node_data


def _calculate_visual_pruning_layout(tree_data):
    """
    Calculate layout for visual pruning mode.
    Vertical position based on impurity decrease.
    """
    layout = {}
    
    # Step 1: Calculate horizontal positions (same as before)
    _calculate_horizontal_positions(tree_data, layout, x=0.5, level=0)
    
    # Step 2: Calculate vertical positions based on impurity decrease
    root_impurity = tree_data['impurity']
    _calculate_vertical_positions_vp(tree_data, layout, parent_y=0, 
                                      parent_impurity=root_impurity,
                                      parent_proportion=tree_data['proportion'],
                                      root_impurity=root_impurity)
    
    # Step 3: Normalize vertical positions to fit screen
    _normalize_vertical_positions(layout)
    
    # Step 4: Calculate V(T) levels
    vt_levels = _calculate_vt_levels(tree_data, layout, root_impurity)
    
    return {
        'nodes': layout,
        'vt_levels': vt_levels,
        'root_impurity': root_impurity
    }


def _calculate_horizontal_positions(node, layout, x, level, l_r=1.0):
    """Calculate horizontal positions (unchanged from original)."""
    if node is None:
        return
    
    node_id = node['id']
    layout[node_id] = {'x': x, 'level': level}
    
    children = node.get('children', [])
    if len(children) == 0:
        return
    
    # Calculate spacing
    n_children = len(children)
    if n_children == 1:
        child_x = x
    else:
        # Split space
        left_x = x - l_r / (2 ** (level + 1))
        right_x = x + l_r / (2 ** (level + 1))
        
        for i, child in enumerate(children):
            if child:
                child_x = left_x if i == 0 else right_x
                _calculate_horizontal_positions(child, layout, child_x, level + 1, l_r)
        return
    
    # Single child case
    for child in children:
        if child:
            _calculate_horizontal_positions(child, layout, child_x, level + 1, l_r)


def _calculate_vertical_positions_vp(node, layout, parent_y, parent_impurity, 
                                     parent_proportion, root_impurity):
    """
    Calculate vertical positions based on impurity decrease.
    Formula: Δ(t→z) = i(t)·p(t) - i(z)·p(z)
    """
    if node is None:
        return
    
    node_id = node['id']
    
    # Calculate decrease
    weighted_parent = parent_impurity * parent_proportion
    weighted_current = node['impurity'] * node['proportion']
    decrease = weighted_parent - weighted_current
    
    # Update y position (increase downward)
    current_y = parent_y + decrease
    layout[node_id]['y'] = current_y
    layout[node_id]['decrease'] = decrease
    layout[node_id]['impurity'] = node['impurity']
    layout[node_id]['proportion'] = node['proportion']
    
    # Recurse to children
    for child in node.get('children', []):
        if child:
            _calculate_vertical_positions_vp(
                child, layout, current_y, 
                node['impurity'], node['proportion'],
                root_impurity
            )


def _normalize_vertical_positions(layout):
    """Normalize y positions to fit in [0, 1] range."""
    if not layout:
        return
    
    y_values = [data['y'] for data in layout.values()]
    y_min = min(y_values)
    y_max = max(y_values)
    
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0
    
    # Normalize to [0.1, 0.9] to leave margin
    for node_id in layout:
        layout[node_id]['y'] = 0.1 + 0.8 * (layout[node_id]['y'] - y_min) / y_range


def _calculate_vt_levels(tree_data, layout, root_impurity):
    """
    Calculate V(T) and Vt(T) for each cutting level.
    
    Returns list of levels with:
    - y: vertical position
    - V_T: weighted impurity if cut at this level
    - Vt_T: relative impurity
    - nodes: nodes that would become leaves
    """
    levels = []
    
    # Group nodes by y position
    y_positions = {}
    for node_id, node_layout in layout.items():
        y = node_layout['y']
        if y not in y_positions:
            y_positions[y] = []
        y_positions[y].append(node_id)
    
    # For each unique y, calculate V(T)
    for y in sorted(y_positions.keys()):
        # Find all nodes at or above this level that would become leaves
        leaf_nodes = _find_cut_leaves(tree_data, layout, y)
        
        # Calculate V(T) = Σ i(h) * p(h)
        V_T = sum(layout[nid]['impurity'] * layout[nid]['proportion'] 
                  for nid in leaf_nodes)
        
        # Calculate Vt(T) for the first node at this level
        nodes_at_level = y_positions[y]
        if nodes_at_level:
            rep_node_id = nodes_at_level[0]
            rep_i = layout[rep_node_id]['impurity']
            rep_p = layout[rep_node_id]['proportion']
            Vt_T = (root_impurity - rep_p * rep_i) / root_impurity if root_impurity > 0 else 0.0
        else:
            Vt_T = 0.0
        
        levels.append({
            'y': y,
            'V_T': V_T,
            'Vt_T': Vt_T,
            'nodes': leaf_nodes
        })
    
    # Identify pruning line (first time Vt(T) - Vt(Z) < 0.01)
    pruning_y = None
    for i in range(len(levels) - 1):
        delta_vt = levels[i]['Vt_T'] - levels[i+1]['Vt_T']
        if delta_vt < 0.01:
            pruning_y = levels[i+1]['y']
            break
    
    return {
        'levels': levels,
        'pruning_y': pruning_y
    }


def _find_cut_leaves(tree_data, layout, cut_y):
    """
    Find all nodes that would become leaves if we cut at y=cut_y.
    """
    leaves = []
    
    def traverse(node):
        if node is None:
            return
        
        node_id = node['id']
        node_y = layout[node_id]['y']
        
        # If this node is below cut, it would be pruned
        if node_y > cut_y:
            return
        
        # If this node is at or above cut, but its children are below
        children = node.get('children', [])
        has_children_below = any(
            child and layout[child['id']]['y'] > cut_y 
            for child in children
        )
        
        if has_children_below or len(children) == 0:
            # This becomes a leaf
            leaves.append(node_id)
        else:
            # Check children
            for child in children:
                if child:
                    traverse(child)
    
    traverse(tree_data)
    return leaves


def _calculate_standard_layout(tree_data):
    """Calculate standard layout (non-visual-pruning)."""
    layout = {}
    _calculate_horizontal_positions(tree_data, layout, x=0.5, level=0)
    
    # Standard vertical positioning by level
    for node_id, data in layout.items():
        data['y'] = data['level'] * 0.15
    
    return {'nodes': layout, 'vt_levels': None, 'root_impurity': tree_data['impurity']}


def _generate_html(tree_data, layout, title, visual_pruning):
    """Generate HTML with D3.js visualization - ORIGINAL VERSION."""
    
    nodes_layout = layout['nodes']
    vt_levels = layout.get('vt_levels')
    
    # Helper function to convert numpy types to native Python types
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(val) for key, val in obj.items()}
        else:
            return obj
    
    # slbt/plotting.py - Fix nella funzione flatten_tree dentro _generate_html

    def flatten_tree(node, depth=0):
        if node is None:
            return []
        
        node_id = node['id']
        layout_data = nodes_layout[node_id]
        
        nodes = [{
            'id': int(node_id),
            'depth': depth,
            'x': float(layout_data['x']),
            'y': float(layout_data.get('y', depth * 0.15)),
            'is_leaf': bool(node['is_leaf']),
            'impurity': float(node['impurity']) if node['impurity'] is not None else 0.0,
            'proportion': float(node['proportion']),
            'n_samples': int(node['n_samples']),
            'distribution': convert_to_native(node['distribution']),
            'feature': str(node['feature']) if node['feature'] is not None else None,
            'threshold': convert_to_native(node['threshold']),
            'prediction': convert_to_native(node['prediction']),
            'gpi': float(node['gpi']) if node['gpi'] is not None else None,
            'ppi': float(node['ppi']) if node['ppi'] is not None else None,
            'LIFT_1': convert_to_native(node['LIFT_1']),
            'LIFT_2': convert_to_native(node['LIFT_2']),
            'GCR': convert_to_native(node['GCR'])  # ← Fix: usa convert_to_native invece di float()
        }]
        
        for child in node.get('children', []):
            if child:
                nodes.extend(flatten_tree(child, depth + 1))
        
        return nodes
    
    # Flatten tree to links
    def get_links(node):
        if node is None:
            return []
        
        links = []
        for child in node.get('children', []):
            if child:
                links.append({
                    'source': int(node['id']),
                    'target': int(child['id'])
                })
                links.extend(get_links(child))
        
        return links
    
    nodes_list = flatten_tree(tree_data)
    links_list = get_links(tree_data)
    
    # Convert to JSON
    nodes_json = json.dumps(nodes_list)
    links_json = json.dumps(links_list)
    
    # ORIGINAL HTML TEMPLATE
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        
        h1 {{
            color: #2d3748;
            margin: 0 0 30px 0;
            font-size: 28px;
            font-weight: 600;
        }}
        
        #tree-container {{
            width: 100%;
            overflow-x: auto;
            background: #f7fafc;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .link {{
            fill: none;
            stroke: #cbd5e0;
            stroke-width: 2px;
        }}
        
        .node circle {{
            stroke: #4a5568;
            stroke-width: 2px;
        }}
        
        .node.internal circle {{
            fill: #667eea;
        }}
        
        .node.leaf circle {{
            fill: #48bb78;
        }}
        
        .node text {{
            font-size: 11px;
            font-weight: 600;
            fill: #2d3748;
            text-anchor: middle;
            pointer-events: none;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(26, 32, 44, 0.95);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            max-width: 300px;
            z-index: 1000;
        }}
        
        .tooltip-title {{
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #90cdf4;
        }}
        
        .tooltip-section {{
            margin: 8px 0;
            padding: 8px 0;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        
        .tooltip-label {{
            color: #a0aec0;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .tooltip-value {{
            color: white;
            font-weight: 600;
        }}
        
        svg {{
            overflow: visible;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div id="tree-container"></div>
    </div>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const data = {{
            nodes: {nodes_json},
            links: {links_json}
        }};

        // Configuration
        const margin = {{top: 40, right: 120, bottom: 40, left: 120}};
        const width = 1200;
        const height = 600;

        // Create SVG
        const svg = d3.select("#tree-container")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

        const g = svg.append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, height]);

        // Draw links
        const link = g.selectAll(".link")
            .data(data.links)
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("x1", d => {{
                const source = data.nodes.find(n => n.id === d.source);
                return xScale(source.x);
            }})
            .attr("y1", d => {{
                const source = data.nodes.find(n => n.id === d.source);
                return yScale(source.y);
            }})
            .attr("x2", d => {{
                const target = data.nodes.find(n => n.id === d.target);
                return xScale(target.x);
            }})
            .attr("y2", d => {{
                const target = data.nodes.find(n => n.id === d.target);
                return yScale(target.y);
            }});

        // Draw nodes
        const node = g.selectAll(".node")
            .data(data.nodes)
            .enter()
            .append("g")
            .attr("class", d => d.is_leaf ? "node leaf" : "node internal")
            .attr("transform", d => `translate(${{xScale(d.x)}},${{yScale(d.y)}})`);

        // Node circles (or pie charts for distribution)
        node.each(function(d) {{
            const nodeGroup = d3.select(this);
            
            if (d.distribution && d.distribution.length > 0) {{
                // Pie chart
                const radius = 20;
                const pie = d3.pie().sort(null);
                const arc = d3.arc()
                    .innerRadius(0)
                    .outerRadius(radius);
                
                const total = d3.sum(d.distribution);
                const pieData = pie(d.distribution.map(v => v / total));
                
                const colors = d3.schemeSet3;
                
                nodeGroup.selectAll("path")
                    .data(pieData)
                    .enter()
                    .append("path")
                    .attr("d", arc)
                    .attr("fill", (p, i) => colors[i % colors.length])
                    .attr("stroke", "#4a5568")
                    .attr("stroke-width", 2);
            }} else {{
                // Simple circle
                nodeGroup.append("circle")
                    .attr("r", 15);
            }}
        }});

        // Node labels
        node.append("text")
            .attr("dy", -25)
            .text(d => `Node ${{d.id}}`);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        node.on("mouseover", function(event, d) {{
            const tooltipContent = `
                <div class="tooltip-title">Node ${{d.id}}</div>
                <div class="tooltip-section">
                    <div class="tooltip-label">Samples</div>
                    <div class="tooltip-value">${{d.n_samples}}</div>
                </div>
                <div class="tooltip-section">
                    <div class="tooltip-label">Impurity</div>
                    <div class="tooltip-value">${{d.impurity.toFixed(4)}}</div>
                </div>
                <div class="tooltip-section">
                    <div class="tooltip-label">Proportion</div>
                    <div class="tooltip-value">${{(d.proportion * 100).toFixed(2)}}%</div>
                </div>
                ${{d.gpi !== null ? `
                <div class="tooltip-section">
                    <div class="tooltip-label">GPI</div>
                    <div class="tooltip-value">${{d.gpi.toFixed(4)}}</div>
                </div>` : ''}}
                ${{d.ppi !== null ? `
                <div class="tooltip-section">
                    <div class="tooltip-label">PPI</div>
                    <div class="tooltip-value">${{d.ppi.toFixed(4)}}</div>
                </div>` : ''}}
                ${{d.is_leaf ? `
                <div class="tooltip-section">
                    <div class="tooltip-label">Prediction</div>
                    <div class="tooltip-value">${{d.prediction}}</div>
                </div>` : `
                <div class="tooltip-section">
                    <div class="tooltip-label">Split Feature</div>
                    <div class="tooltip-value">${{d.feature}}</div>
                </div>
                <div class="tooltip-section">
                    <div class="tooltip-label">Threshold</div>
                    <div class="tooltip-value">${{JSON.stringify(d.threshold)}}</div>
                </div>`}}
                ${{d.distribution ? `
                <div class="tooltip-section">
                    <div class="tooltip-label">Distribution</div>
                    <div class="tooltip-value">${{d.distribution.map((v, i) => `Class ${{i}}: ${{v}}`).join('<br>')}}</div>
                </div>` : ''}}
            `;
            
            tooltip
                .html(tooltipContent)
                .style("opacity", 1)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 15) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.style("opacity", 0);
        }});
    </script>
</body>
</html>'''
    
    return html