# plot_by_map_type.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import os

def ensure_dir(directory):
    """Создаёт директорию, если её нет"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_results_for_folder(folder_path):
    """Загружает результаты всех моделей из конкретной папки"""
    results = defaultdict(list)
    
    if not folder_path.exists():
        print(f"⚠️ Папка не найдена: {folder_path}")
        return results
    
    json_files = list(folder_path.glob("*.json"))
    print(f"📁 {folder_path.name}: найдено {len(json_files)} JSON файлов")
    
    for json_file in json_files:
        model_name = json_file.stem
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            num_agents = entry['env_grid_search']['num_agents']
            results[model_name].append({
                'num_agents': num_agents,
                'CSR': entry['metrics']['CSR'],
                'SoC': entry['metrics']['SoC'],
                'runtime': entry['metrics']['runtime'],
                'ISR': entry['metrics']['ISR']
            })
    
    return results

# ФИКСИРОВАННЫЕ ЦВЕТА И СТИЛИ ДЛЯ КАЖДОЙ МОДЕЛИ
MODEL_STYLES = {
    'MAPF-GPT-2M': {
        'color': '#e74c3c',      # красный
        'marker': 'o',
        'linestyle': '-',
        'label': '2M (baseline)'
    },
    'MAPF-GPT-2M-L4': {
        'color': '#f39c12',      # оранжевый
        'marker': 's',
        'linestyle': '--',
        'label': '2M-L4 (trim)'
    },
    'MAPF-GPT-2M-L4-ft': {
        'color': '#2ecc71',      # зелёный
        'marker': '^',
        'linestyle': '-.',
        'label': '2M-L4 (ft)'
    },
    'MAPF-GPT-2M-L3': {
        'color': '#9b59b6',      # фиолетовый
        'marker': 'D',
        'linestyle': ':',
        'label': '2M-L3 (trim)'
    },
    'MAPF-GPT-2M-L3-ft': {
        'color': '#3498db',      # синий
        'marker': 'v',
        'linestyle': '-',
        'label': '2M-L3 (ft)'
    },
    'MAPF-GPT-2M-int8': {
        'color': '#1abc9c',      # бирюзовый
        'marker': '<',
        'linestyle': '--',
        'label': '2M-int8 (quant)'
    }
}

def plot_metric_for_folder(results, map_name, metric_name, title, ylabel, output_dir):
    """Строит график для заданной метрики для всех моделей на одном типе карт"""
    plt.figure(figsize=(10, 6))
    
    plotted = 0
    
    for model_name, entries in results.items():
        if model_name not in MODEL_STYLES:
            print(f"⚠️ Нет стиля для модели: {model_name}")
            continue
        
        style = MODEL_STYLES[model_name]
        
        # Группируем по num_agents и усредняем
        agents_data = defaultdict(lambda: {'values': [], 'count': 0})
        for entry in entries:
            na = entry['num_agents']
            agents_data[na]['values'].append(entry[metric_name])
            agents_data[na]['count'] += 1
        
        # Сортируем по количеству агентов
        sorted_agents = sorted(agents_data.keys())
        means = [np.mean(agents_data[a]['values']) for a in sorted_agents]
        stds = [np.std(agents_data[a]['values']) for a in sorted_agents]
        
        plt.errorbar(sorted_agents, means, yerr=stds, 
                    marker=style['marker'], capsize=5, 
                    label=style['label'], 
                    color=style['color'], linewidth=2, markersize=8,
                    linestyle=style['linestyle'])
        plotted += 1
    
    if plotted == 0:
        print(f"⚠️ Нет данных для {map_name} - {metric_name}")
        plt.close()
        return
    
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{map_name}: {title}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f'{map_name}_{metric_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сохранён: {output_path}")

def main():
    base_path = Path("eval_configs")
    output_dir = Path("plots/by_map")
    ensure_dir(output_dir)
    
    # Карты для анализа
    map_folders = {
        "Random": base_path / "01-random",
        "Mazes": base_path / "02-mazes", 
        "Warehouse": base_path / "03-warehouse",
        "MovingAI": base_path  / "04-movingai",
        "Puzzles": base_path / "05-puzzles"
    }
    
    metrics = [
        ('CSR', 'CSR vs Number of Agents', 'CSR (Conflict-free Success Rate)'),
        ('SoC', 'Sum of Costs vs Number of Agents', 'SoC (Sum of Costs)'),
        ('runtime', 'Runtime vs Number of Agents', 'Runtime (seconds)')
    ]
    
    print("📂 Загрузка результатов...\n")
    
    for map_name, folder_path in map_folders.items():
        print(f"\n{'='*50}")
        print(f"📊 Обработка карт: {map_name}")
        print(f"{'='*50}")
        
        # Загружаем результаты для этого типа карт
        results = load_results_for_folder(folder_path)
        
        if not results:
            print(f"⚠️ Нет данных для {map_name}")
            continue
        
        # Показываем какие модели найдены
        models_found = [MODEL_STYLES.get(m, {}).get('label', m) for m in results.keys()]
        print(f"📋 Найдены модели: {', '.join(models_found)}")
        
        # Строим графики для каждой метрики
        for metric_name, title, ylabel in metrics:
            plot_metric_for_folder(results, map_name, metric_name, title, ylabel, output_dir)
    
    print(f"\n{'='*50}")
    print(f"✅ Все графики сохранены в папку: {output_dir.resolve()}")

if __name__ == "__main__":
    main()