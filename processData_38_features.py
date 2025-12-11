import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
import pandas as pd
from tqdm import tqdm
from collections import Counter

# 数据处理流程说明（增强版 - 密度波形特征 + 原始vasp结构特征 + 晶体整体特征）：
# 1. 从 data/split/fail 和 data/split/success 文件夹中读取 max 和 min 子文件夹中的同名 .vasp 文件
# 2. 对每对 max/min 文件提取7个特征，然后计算特征比值（max特征/min特征）得到7个比值特征
# 3. 从原始 vasp 文件中提取21个密度分布波形特征
# 4. 从原始 vasp 文件中提取7个结构特征
# 5. 从原始 vasp 文件中提取3个晶体整体特征
# 6. 将7个比值特征、21个密度波形特征、7个原始结构特征和3个晶体整体特征整合，得到38个特征
# 7. fail 文件夹的样本标签为 0，success 文件夹的样本标签为 1
# 8. 最终生成包含38个特征的 CSV 文件（features_38_enhanced.csv）

# 导入密度特征提取所需的库
import sys
sys.path.append('.')
from scipy.signal import argrelextrema
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from ase.io import read

# 定义密度分布
fenbu = np.array([0,0.0011,0.0044,0.01,0.0178,0.0278,0.04,0.0544,0.0711,0.09,0.1111,0.1344,0.16,0.1878,0.2178,0.25,0.2844,0.3211,0.36,0.4011,0.4444,0.49,0.5378,0.5878,0.64,0.6944,0.7511,0.81,0.8711,0.9344,1,0.9344,0.8711,0.81,0.7511,0.6944,0.64,0.5878,0.5378,0.49,0.4444,0.4011,0.36,0.3211,0.2844,0.25,0.2178,0.1878,0.16,0.1344,0.1111,0.09,0.0711,0.0544,0.04,0.0278,0.0178,0.01,0.0044,0.0011,0])

# 定义原子属性的获取函数
def get_atom_features(structure):
    """获取原子的电负性和原子半径特征"""
    features = []
    for site in structure:
        element = Element(site.species_string)
        feature = [element.X, element.atomic_radius]
        features.append(feature)
    return np.array(features, dtype=np.float32)

# 定义边的构建函数
def get_edges(structure, cutoff_factor=1.2):
    """使用距离阈值判断原子间连接，基于共价半径"""
    edges = []
    edge_distances = []
    edge_types = []
    edge_weights = []
    
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            distance = structure[i].distance(structure[j])
            elem_i = Element(structure[i].species_string)
            elem_j = Element(structure[j].species_string)
            covalent_radius_sum = (elem_i.atomic_radius + elem_j.atomic_radius) * cutoff_factor
            if distance <= covalent_radius_sum:
                edges.append([i, j])
                edges.append([j, i])
                edge_distances.append(distance)
                edge_distances.append(distance)
                electroneg_diff = abs(elem_i.X - elem_j.X)
                bond_type = 1 if electroneg_diff > 1.7 else 0
                edge_types.append(bond_type)
                edge_types.append(bond_type)
                weight = 1.0 / (distance + 1e-6)
                edge_weights.append(weight)
                edge_weights.append(weight)

    edges = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_distances = np.array(edge_distances, dtype=np.float32) if edge_distances else np.zeros(0, dtype=np.float32)
    edge_types = np.array(edge_types, dtype=np.float32) if edge_types else np.zeros(0, dtype=np.float32)
    edge_weights = np.array(edge_weights, dtype=np.float32) if edge_weights else np.zeros(0, dtype=np.float32)
    
    return edges, edge_distances, edge_types, edge_weights

# 定义聚合特征生成函数（7个特征）
def extract_aggregated_features(structure):
    """从晶体结构中提取聚合特征，总共7个特征"""
    atom_features = get_atom_features(structure)
    edges, edge_distances, edge_types, edge_weights = get_edges(structure)
    
    features = []
    features.append(np.mean(atom_features[:, 0]))  # node_electronegativity
    features.append(np.mean(atom_features[:, 1]))  # node_atomic_radius
    features.append(np.mean(edge_distances) if len(edge_distances) > 0 else 0.0)  # edge_distance
    features.append(np.mean(edge_types) if len(edge_types) > 0 else 0.0)  # edge_bond_type
    features.append(np.mean(edge_weights) if len(edge_weights) > 0 else 0.0)  # edge_weight
    num_edges = len(edge_distances) // 2
    features.append(num_edges)  # n_edges
    num_atoms = len(structure)
    features.append(num_edges / num_atoms if num_atoms > 0 else 0)  # edge_node_ratio
    
    return np.array(features, dtype=np.float32)

def no_sys(filename):
    """计算z方向密度分布"""
    sur_atom = read(filename, format='vasp')
    num_atom = np.shape(sur_atom.positions)
    z_max = np.max(sur_atom.positions[:,2])
    z_min = np.min(sur_atom.positions[:,2])
    z_max = round(z_max+3,1)
    z_min = round(z_min-3,1)
    z_d = round((z_max-z_min)*10)
    labelx = 0
    z_p = np.zeros([z_d+1,2])
    for i in np.arange(z_min,z_max,0.1).round(1):
        z_p[labelx,0] = round(i,1)       
        labelx = labelx+1
    
    for i in range(num_atom[0]):
        zz = round(sur_atom.positions[i,2],1)
        lb = np.where(z_p[:,0] == zz)
        for n in range(61):
            z_p[lb[0]+n-31,1] = z_p[lb[0]+n-31,1]+fenbu[n]
    return z_p

def extract_waveform_features(densities):
    """
    从密度波形中提取波形特征（删除了11个特征，保留17个）
    删除的特征：n_peaks, n_valleys, peak_valley_ratio, peak_mean, peak_std, 
               peak_distance_mean, valley_mean, valley_std, density_mean, 
               low_high_freq_ratio, spectral_centroid
    """
    features = {}
    
    # 峰谷统计特征 - 只保留 peak_range, peak_distance_std
    local_maxima = argrelextrema(densities, np.greater)[0]
    
    if len(local_maxima) > 0:
        peak_values = densities[local_maxima]
        features['peak_range'] = np.max(peak_values) - np.min(peak_values)
        if len(local_maxima) > 1:
            peak_distances = np.diff(local_maxima)
            features['peak_distance_std'] = np.std(peak_distances)
        else:
            features['peak_distance_std'] = 0.0
    else:
        features['peak_range'] = 0.0
        features['peak_distance_std'] = 0.0
    
    # 谷值统计 - 全部删除（n_valleys, valley_mean, valley_std都删除了）
    
    # 波形形态特征 - 保留全部（删除density_mean）
    features['density_skewness'] = skew(densities)
    features['density_kurtosis'] = kurtosis(densities)
    features['density_std'] = np.std(densities)
    features['density_range'] = np.max(densities) - np.min(densities)
    features['density_energy'] = np.sum(densities ** 2)
    
    density_normalized = densities / (np.sum(densities) + 1e-10)
    features['density_entropy'] = -np.sum(density_normalized * np.log(density_normalized + 1e-10))
    
    # 导数特征 - 保留全部
    first_derivative = np.diff(densities)
    features['first_derivative_mean'] = np.mean(np.abs(first_derivative))
    features['first_derivative_std'] = np.std(first_derivative)
    features['first_derivative_max'] = np.max(np.abs(first_derivative))
    
    if len(first_derivative) > 1:
        second_derivative = np.diff(first_derivative)
        features['second_derivative_mean'] = np.mean(np.abs(second_derivative))
        features['second_derivative_std'] = np.std(second_derivative)
    else:
        features['second_derivative_mean'] = 0.0
        features['second_derivative_std'] = 0.0
    
    # 频域特征 - 只保留 fft_max_amplitude, fft_mean_amplitude（删除low_high_freq_ratio, spectral_centroid）
    if len(densities) > 4:
        fft_values = np.abs(fft(densities))
        fft_values = fft_values[:len(fft_values)//2]
        if len(fft_values) > 1:
            features['fft_max_amplitude'] = np.max(fft_values[1:])
            features['fft_mean_amplitude'] = np.mean(fft_values[1:])
        else:
            features['fft_max_amplitude'] = 0.0
            features['fft_mean_amplitude'] = 0.0
    else:
        features['fft_max_amplitude'] = 0.0
        features['fft_mean_amplitude'] = 0.0
    
    # 周期性特征 - 保留全部（periodicity_lag, periodicity_strength）
    if len(densities) > 10:
        autocorr = np.correlate(densities - np.mean(densities), 
                               densities - np.mean(densities), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        if len(autocorr) > 3:
            peaks_ac = argrelextrema(autocorr[1:], np.greater)[0]
            if len(peaks_ac) > 0:
                features['periodicity_lag'] = peaks_ac[0] + 1
                features['periodicity_strength'] = autocorr[peaks_ac[0] + 1]
            else:
                features['periodicity_lag'] = 0.0
                features['periodicity_strength'] = 0.0
        else:
            features['periodicity_lag'] = 0.0
            features['periodicity_strength'] = 0.0
    else:
        features['periodicity_lag'] = 0.0
        features['periodicity_strength'] = 0.0
    
    return features

def extract_density_features_enhanced(filename):
    """
    从vasp文件中提取增强的密度分布特征（21个特征）：
    - 原始4个特征（最大值、最小值、比值、差值）
    - 17个波形特征
    """
    try:
        z_p_current = no_sys(filename)
        sur_atom = read(filename, format='vasp')
        z_positions = sur_atom.positions[:, 2]
        z_atom_min = z_positions.min()
        z_atom_max = z_positions.max()
        
        all_z_coords = z_p_current[:, 0]
        all_densities = z_p_current[:, 1]
        
        mask_nonzero = all_densities > 0
        z_coords = all_z_coords[mask_nonzero]
        densities = all_densities[mask_nonzero]
        
        max_idx = np.argmax(densities)
        density_max = densities[max_idx]
        z_max = z_coords[max_idx]
        
        search_margin = 0.5
        valid_search_mask = (z_coords >= z_atom_min - search_margin) & (z_coords <= z_atom_max + search_margin)
        
        z_coords_search = z_coords[valid_search_mask]
        densities_search = densities[valid_search_mask]
        search_indices = np.where(valid_search_mask)[0]
        
        local_min_indices_search = argrelextrema(densities_search, np.less)[0]
        
        if len(local_min_indices_search) > 0:
            local_min_indices = search_indices[local_min_indices_search]
            z_local_mins = z_coords[local_min_indices]
            density_local_mins = densities[local_min_indices]
            
            min_density_idx = np.argmin(density_local_mins)
            min_density_value = density_local_mins[min_density_idx]
            
            min_density_mask = density_local_mins == min_density_value
            candidate_indices = local_min_indices[min_density_mask]
            candidate_z = z_local_mins[min_density_mask]
            
            if len(candidate_indices) == 1:
                chosen_idx = candidate_indices[0]
            else:
                distances = np.abs(candidate_z - z_max)
                closest_among_mins = np.argmin(distances)
                chosen_idx = candidate_indices[closest_among_mins]
            
            density_min = densities[chosen_idx]
        else:
            min_idx_search = np.argmin(densities_search)
            chosen_idx = search_indices[min_idx_search]
            density_min = densities[chosen_idx]
        
        # 4个基础特征
        basic_features = [
            density_max,
            density_min,
            density_max / density_min if density_min != 0 else 1.0,
            density_max - density_min
        ]
        
        # 17个波形特征
        waveform_features_dict = extract_waveform_features(densities_search)
        
        waveform_features_list = [
            waveform_features_dict['peak_range'],
            waveform_features_dict['peak_distance_std'],
            waveform_features_dict['density_skewness'],
            waveform_features_dict['density_kurtosis'],
            waveform_features_dict['density_std'],
            waveform_features_dict['density_range'],
            waveform_features_dict['density_energy'],
            waveform_features_dict['density_entropy'],
            waveform_features_dict['first_derivative_mean'],
            waveform_features_dict['first_derivative_std'],
            waveform_features_dict['first_derivative_max'],
            waveform_features_dict['second_derivative_mean'],
            waveform_features_dict['second_derivative_std'],
            waveform_features_dict['fft_max_amplitude'],
            waveform_features_dict['fft_mean_amplitude'],
            waveform_features_dict['periodicity_lag'],
            waveform_features_dict['periodicity_strength']
        ]
        
        all_features = basic_features + waveform_features_list
        return np.array(all_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting density features from {filename}: {e}")
        return None

# ============ 晶体整体结构特征提取（从processData3-2-3.ipynb复制） ============

def split_structure_into_layers(structure, z_tolerance=0.5):
    """将晶体结构沿z轴分成多个层"""
    z_coords = np.array([site.coords[2] for site in structure])
    sorted_indices = np.argsort(z_coords)
    sorted_z = z_coords[sorted_indices]
    
    layers = []
    layer_z_positions = []
    current_layer = [sorted_indices[0]]
    current_z = sorted_z[0]
    
    for i in range(1, len(sorted_indices)):
        if sorted_z[i] - current_z <= z_tolerance:
            current_layer.append(sorted_indices[i])
        else:
            layers.append(current_layer)
            layer_z_positions.append(np.mean([z_coords[idx] for idx in current_layer]))
            current_layer = [sorted_indices[i]]
            current_z = sorted_z[i]
    
    if current_layer:
        layers.append(current_layer)
        layer_z_positions.append(np.mean([z_coords[idx] for idx in current_layer]))
    
    return layers, layer_z_positions

def get_layer_composition(structure, layer_indices):
    """获取某一层的原子种类组成"""
    composition = Counter()
    for idx in layer_indices:
        elem = str(structure[idx].specie)
        composition[elem] += 1
    return composition

def calculate_composition_similarity(comp1, comp2):
    """计算两个原子组成的相似度（归一化后的余弦相似度）"""
    all_elements = set(comp1.keys()) | set(comp2.keys())
    if len(all_elements) == 0:
        return 1.0
    
    total1 = sum(comp1.values())
    total2 = sum(comp2.values())
    
    vec1 = np.array([comp1.get(e, 0) / max(total1, 1) for e in sorted(all_elements)])
    vec2 = np.array([comp2.get(e, 0) / max(total2, 1) for e in sorted(all_elements)])
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)

def extract_layer_composition_consistency(structure, z_tolerance=0.5):
    """特征1: 计算晶体各层原子种类的一致性"""
    try:
        layers, _ = split_structure_into_layers(structure, z_tolerance)
        if len(layers) <= 1:
            return 1.0
        
        compositions = [get_layer_composition(structure, layer) for layer in layers]
        similarities = []
        for i in range(len(compositions)):
            for j in range(i + 1, len(compositions)):
                sim = calculate_composition_similarity(compositions[i], compositions[j])
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 1.0
        return float(np.mean(similarities))
    except Exception as e:
        print(f"Error in layer composition consistency: {e}")
        return 0.0

def extract_z_periodicity(structure, z_tolerance=0.5, max_period=5):
    """特征2: 检测晶体在z轴方向上的周期性"""
    try:
        layers, layer_z = split_structure_into_layers(structure, z_tolerance)
        if len(layers) <= 2:
            return 0.0
        
        compositions = [get_layer_composition(structure, layer) for layer in layers]
        best_periodicity = 0.0
        
        for period in range(1, min(max_period + 1, len(layers) // 2 + 1)):
            period_similarities = []
            for i in range(len(layers) - period):
                sim = calculate_composition_similarity(compositions[i], compositions[i + period])
                period_similarities.append(sim)
            
            if len(period_similarities) > 0:
                avg_similarity = np.mean(period_similarities)
                coverage = len(period_similarities) / max(len(layers) - period, 1)
                periodicity_score = avg_similarity * coverage
                best_periodicity = max(best_periodicity, periodicity_score)
        
        return float(best_periodicity)
    except Exception as e:
        print(f"Error in z periodicity: {e}")
        return 0.0

def extract_z_symmetry(structure, z_tolerance=0.5):
    """特征3: 检测晶体在z轴方向上的对称性"""
    try:
        layers, layer_z = split_structure_into_layers(structure, z_tolerance)
        if len(layers) <= 1:
            return 1.0
        
        compositions = [get_layer_composition(structure, layer) for layer in layers]
        n_layers = len(compositions)
        symmetry_scores = []
        
        for i in range(n_layers // 2):
            sim = calculate_composition_similarity(compositions[i], compositions[-(i + 1)])
            symmetry_scores.append(sim)
        
        if len(symmetry_scores) == 0:
            return 1.0
        return float(np.mean(symmetry_scores))
    except Exception as e:
        print(f"Error in z symmetry: {e}")
        return 0.0

def extract_crystal_global_features(structure, z_tolerance=0.5):
    """提取晶体整体特征（3个特征）"""
    features = []
    features.append(extract_layer_composition_consistency(structure, z_tolerance))
    features.append(extract_z_periodicity(structure, z_tolerance, max_period=5))
    features.append(extract_z_symmetry(structure, z_tolerance))
    return np.array(features, dtype=np.float32)

# ============ 主函数 ============

def find_matching_files_with_vasp(fail_dir, success_dir, vasp_fail_dir, vasp_success_dir):
    """找到 fail 和 success 文件夹中 max 和 min 子文件夹里的匹配文件对"""
    file_pairs = []
    
    # 处理 fail 文件夹 (label=0)
    fail_max_dir = os.path.join(fail_dir, 'max')
    fail_min_dir = os.path.join(fail_dir, 'min')
    
    if os.path.exists(fail_max_dir) and os.path.exists(fail_min_dir):
        max_files = {f: os.path.join(fail_max_dir, f) for f in os.listdir(fail_max_dir) if f.endswith('.vasp')}
        min_files = {f: os.path.join(fail_min_dir, f) for f in os.listdir(fail_min_dir) if f.endswith('.vasp')}
        common_files = set(max_files.keys()) & set(min_files.keys())
        
        matched_count = 0
        for filename in common_files:
            vasp_path = os.path.join(vasp_fail_dir, filename)
            if os.path.exists(vasp_path):
                file_pairs.append((max_files[filename], min_files[filename], vasp_path, 0, filename))
                matched_count += 1
        print(f"Fail folder: Found {len(common_files)} matching max/min pairs, {matched_count} with vasp files")
    
    # 处理 success 文件夹 (label=1)
    success_max_dir = os.path.join(success_dir, 'max')
    success_min_dir = os.path.join(success_dir, 'min')
    
    if os.path.exists(success_max_dir) and os.path.exists(success_min_dir):
        max_files = {f: os.path.join(success_max_dir, f) for f in os.listdir(success_max_dir) if f.endswith('.vasp')}
        min_files = {f: os.path.join(success_min_dir, f) for f in os.listdir(success_min_dir) if f.endswith('.vasp')}
        common_files = set(max_files.keys()) & set(min_files.keys())
        
        matched_count = 0
        for filename in common_files:
            vasp_path = os.path.join(vasp_success_dir, filename)
            if os.path.exists(vasp_path):
                file_pairs.append((max_files[filename], min_files[filename], vasp_path, 1, filename))
                matched_count += 1
        print(f"Success folder: Found {len(common_files)} matching max/min pairs, {matched_count} with vasp files")
    
    return file_pairs

def process_file_pair_with_all_features(max_path, min_path, vasp_path, label, filename):
    """
    处理一对 max/min VASP 文件和对应的原始vasp文件
    返回: (combined_features, label, filename) 或 None
    combined_features包含：7个比值特征 + 21个密度特征 + 7个原始vasp结构特征 + 3个晶体整体特征 = 38个特征
    """
    try:
        # 处理 max 文件
        max_structure = Structure.from_file(max_path)
        if not max_structure.is_valid():
            print(f"Warning: Invalid max structure in {filename}, skipping.")
            return None
        max_features = extract_aggregated_features(max_structure)
        
        # 处理 min 文件
        min_structure = Structure.from_file(min_path)
        if not min_structure.is_valid():
            print(f"Warning: Invalid min structure in {filename}, skipping.")
            return None
        min_features = extract_aggregated_features(min_structure)
        
        # 计算特征比值 (max / min)
        ratio_features = np.divide(max_features, min_features, 
                                  out=np.ones_like(max_features), 
                                  where=min_features!=0)
        ratio_features = np.nan_to_num(ratio_features, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 提取原始vasp文件的增强密度特征（21个：4基础 + 17波形）
        density_features = extract_density_features_enhanced(vasp_path)
        if density_features is None:
            print(f"Warning: Failed to extract density features from {filename}, skipping.")
            return None
        
        # 提取原始vasp文件的结构特征（7个）
        vasp_structure = Structure.from_file(vasp_path)
        if not vasp_structure.is_valid():
            print(f"Warning: Invalid vasp structure in {filename}, skipping.")
            return None
        vasp_structure_features = extract_aggregated_features(vasp_structure)
        
        # 提取原始vasp文件的晶体整体特征（3个）
        crystal_global_features = extract_crystal_global_features(vasp_structure, z_tolerance=0.5)
        
        # 整合特征：7 + 21 + 7 + 3 = 38个特征
        combined_features = np.concatenate([ratio_features, density_features, vasp_structure_features, crystal_global_features])
        
        return combined_features, label, filename
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def process_dataset_with_all_features(fail_dir, success_dir, vasp_fail_dir, vasp_success_dir, output_csv_path):
    """处理整个数据集，共38个特征，并保存为CSV文件"""
    file_pairs = find_matching_files_with_vasp(fail_dir, success_dir, vasp_fail_dir, vasp_success_dir)
    print(f"\nTotal matching file pairs found: {len(file_pairs)}")
    
    if len(file_pairs) == 0:
        print("Error: No matching file pairs found!")
        return None
    
    data_list = []
    labels_list = []
    filenames_list = []
    
    for max_path, min_path, vasp_path, label, filename in tqdm(file_pairs, desc="Processing file pairs"):
        result = process_file_pair_with_all_features(max_path, min_path, vasp_path, label, filename)
        if result is not None:
            combined_features, label, filename = result
            data_list.append(combined_features)
            labels_list.append(label)
            filenames_list.append(filename)
    
    if len(data_list) == 0:
        print("Error: No valid data processed!")
        return None
    
    X = np.array(data_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    
    print(f"\nProcessed {len(data_list)} valid structure pairs")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    print(f"Number of features per sample: {X.shape[1]}")
    print(f"Number of success samples (label=1): {np.sum(y == 1)}")
    print(f"Number of fail samples (label=0): {np.sum(y == 0)}")
    
    # 创建特征列名（38个特征：7个比值 + 21个密度 + 7个原始vasp结构 + 3个晶体整体）
    feature_names = []
    
    # 前7个特征：比值特征
    ratio_feature_names = [
        'node_electronegativity', 'node_atomic_radius', 'edge_distance',
        'edge_bond_type', 'edge_weight', 'n_edges', 'edge_node_ratio'
    ]
    for feat in ratio_feature_names:
        feature_names.append(f'ratio_{feat}')
    
    # 中间21个特征：密度特征
    density_feature_names = [
        # 基础4个
        'max_density', 'min_density', 'density_ratio', 'density_difference',
        # 峰谷统计2个（删除了n_peaks, n_valleys, peak_valley_ratio, peak_mean, peak_std, peak_distance_mean, valley_mean, valley_std）
        'peak_range', 'peak_distance_std',
        # 形态特征6个（删除了density_mean）
        'density_skewness', 'density_kurtosis', 'density_std',
        'density_range', 'density_energy', 'density_entropy',
        # 导数特征5个
        'first_derivative_mean', 'first_derivative_std', 'first_derivative_max',
        'second_derivative_mean', 'second_derivative_std',
        # 频域特征2个（删除了low_high_freq_ratio, spectral_centroid）
        'fft_max_amplitude', 'fft_mean_amplitude',
        # 周期性特征2个（保留全部）
        'periodicity_lag', 'periodicity_strength'
    ]
    for feat in density_feature_names:
        feature_names.append(feat)
    
    # 接下来7个特征：原始vasp结构特征
    vasp_structure_feature_names = [
        'node_electronegativity', 'node_atomic_radius', 'edge_distance',
        'edge_bond_type', 'edge_weight', 'n_edges', 'edge_node_ratio'
    ]
    for feat in vasp_structure_feature_names:
        feature_names.append(f'vasp_{feat}')
    
    # 最后3个特征：晶体整体特征
    crystal_global_feature_names = [
        'layer_composition_consistency', 'z_periodicity', 'z_symmetry'
    ]
    for feat in crystal_global_feature_names:
        feature_names.append(feat)
    
    if len(feature_names) != 38:
        print(f"Warning: Expected 38 feature names, but got {len(feature_names)}")
    
    if len(feature_names) != X.shape[1]:
        print(f"Warning: Feature names count ({len(feature_names)}) != actual features ({X.shape[1]})")
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df['filename'] = filenames_list
    
    cols = ['filename', 'label'] + feature_names
    df = df[cols]
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\nDataset saved to {output_csv_path}")
    print(f"\nDataFrame preview:")
    print(df.head())
    
    return df

# ============ 执行数据处理 ============
if __name__ == "__main__":
    fail_dir = "data/split/fail"
    success_dir = "data/split/success"
    vasp_fail_dir = "data/vaspFail"
    vasp_success_dir = "data/vaspSuccess"
    output_csv_path = "data/dataset/features_38_enhanced.csv"
    
    print("="*70)
    print("开始处理数据集（38个特征 - 增强版）...")
    print("  - 7个特征来自max/min文件对的比值")
    print("  - 21个特征来自原始vasp文件的密度分布波形")
    print("  - 7个特征来自原始vasp文件的结构特征")
    print("  - 3个特征来自晶体整体特征（层一致性、周期性、对称性）")
    print("="*70)
    
    df = process_dataset_with_all_features(fail_dir, success_dir, 
                                          vasp_fail_dir, vasp_success_dir, 
                                          output_csv_path)
    
    if df is not None:
        print("\n" + "="*70)
        print("数据预处理完成!")
        print(f"数据集形状: {df.shape}")
        print(f"特征数量: {df.shape[1] - 2}")
        print(f"样本数量: {df.shape[0]}")
        print(f"正样本数量 (success): {np.sum(df['label'] == 1)}")
        print(f"负样本数量 (fail): {np.sum(df['label'] == 0)}")
        print("="*70)
        print("\n特征说明:")
        print("  比值特征 (7个):")
        print("    1-7. ratio_* - max/min文件对的7个特征比值")
        print("\n  密度分布特征 (21个):")
        print("    基础特征 (4个):")
        print("      8. max_density - 密度分布最大值")
        print("      9. min_density - 极小值中的最小值")
        print("     10. density_ratio - 最大值/最小值")
        print("     11. density_difference - 最大值-最小值")
        print("\n    峰谷统计特征 (2个):")
        print("     12. peak_range - 峰值范围")
        print("     13. peak_distance_std - 峰间距标准差")
        print("\n    形态特征 (6个):")
        print("     14. density_skewness - 分布偏度")
        print("     15. density_kurtosis - 分布峰度")
        print("     16. density_std - 密度标准差")
        print("     17. density_range - 密度范围")
        print("     18. density_energy - 波形能量")
        print("     19. density_entropy - 波形熵")
        print("\n    导数特征 (5个):")
        print("     20-22. first_derivative_* - 一阶导数统计")
        print("     23-24. second_derivative_* - 二阶导数统计")
        print("\n    频域特征 (2个):")
        print("     25. fft_max_amplitude - 最大频率振幅")
        print("     26. fft_mean_amplitude - 平均频率振幅")
        print("\n    周期性特征 (2个):")
        print("     27. periodicity_lag - 周期性滞后")
        print("     28. periodicity_strength - 周期性强度")
        print("\n  原始vasp结构特征 (7个):")
        print("     29-35. vasp_* - 原始vasp文件的结构特征")
        print("\n  晶体整体特征 (3个):")
        print("     36. layer_composition_consistency - 层原子种类一致性")
        print("     37. z_periodicity - z轴方向周期性强度")
        print("     38. z_symmetry - z轴方向对称性")
        print("="*70)
    else:
        print("\n处理失败，请检查数据文件夹路径和文件是否存在")
