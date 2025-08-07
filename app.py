import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import memory_profiler
import cProfile
import pstats
import io
from collections import defaultdict
import random
import sys
from typing import List, Dict, Any, Tuple
import gc
import psutil
import threading
from dataclasses import dataclass
from contextlib import contextmanager
st.set_page_config(
    page_title="Data Structures Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# st.title("Test Title - If you see this, it's working")



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .performance-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .performance-good {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f3f4;
        border-radius: 5px 5px 0 0;
        color: #1f2937;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PerformanceResult:
    operation: str
    data_structure: str
    size: int
    time_taken: float
    memory_usage: float
    cpu_usage: float
    complexity: str

class DataStructureAnalyzer:
    def __init__(self):
        self.results = []
        self.profiler_results = {}
        
    @contextmanager
    def profile_context(self, operation_name: str):
        """Context manager for profiling operations"""
        profiler = cProfile.Profile()
        process = psutil.Process()
        
        # Start profiling
        profiler.enable()
        start_time = time.time()
        start_cpu = process.cpu_percent()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()
            end_time = time.time()
            end_cpu = process.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Store results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)
            
            self.profiler_results[operation_name] = {
                'profile_data': s.getvalue(),
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'cpu_usage': (start_cpu + end_cpu) / 2
            }

    def benchmark_array_operations(self, size: int) -> List[PerformanceResult]:
        """Benchmark array operations"""
        results = []
        arr = list(range(size))
        
        # Insert operation
        with self.profile_context("array_insert"):
            start_time = time.time()
            temp_arr = arr.copy()
            for i in range(min(1000, size // 10)):
                temp_arr.insert(random.randint(0, len(temp_arr)), random.randint(0, 1000))
            insert_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Insert", "Array", size, insert_time,
            self.profiler_results["array_insert"]["memory_delta"],
            self.profiler_results["array_insert"]["cpu_usage"],
            "O(n)"
        ))
        
        # Search operation
        with self.profile_context("array_search"):
            start_time = time.time()
            for i in range(min(1000, size // 10)):
                target = random.randint(0, size)
                _ = target in arr
            search_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Search", "Array", size, search_time,
            self.profiler_results["array_search"]["memory_delta"],
            self.profiler_results["array_search"]["cpu_usage"],
            "O(n)"
        ))
        
        # Delete operation
        with self.profile_context("array_delete"):
            start_time = time.time()
            temp_arr = arr.copy()
            for i in range(min(1000, size // 10)):
                if temp_arr:
                    temp_arr.pop(random.randint(0, len(temp_arr) - 1))
            delete_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Delete", "Array", size, delete_time,
            self.profiler_results["array_delete"]["memory_delta"],
            self.profiler_results["array_delete"]["cpu_usage"],
            "O(n)"
        ))
        
        return results

    def benchmark_linked_list_operations(self, size: int) -> List[PerformanceResult]:
        """Benchmark linked list operations"""
        results = []
        
        class Node:
            def __init__(self, data):
                self.data = data
                self.next = None
        
        class LinkedList:
            def __init__(self):
                self.head = None
                self.size = 0
                
            def insert(self, data):
                new_node = Node(data)
                new_node.next = self.head
                self.head = new_node
                self.size += 1
                
            def search(self, data):
                current = self.head
                while current:
                    if current.data == data:
                        return True
                    current = current.next
                return False
                
            def delete(self, data):
                if not self.head:
                    return False
                if self.head.data == data:
                    self.head = self.head.next
                    self.size -= 1
                    return True
                current = self.head
                while current.next:
                    if current.next.data == data:
                        current.next = current.next.next
                        self.size -= 1
                        return True
                    current = current.next
                return False
        
        # Create linked list
        linked_list = LinkedList()
        for i in range(size):
            linked_list.insert(i)
        
        # Insert operation
        with self.profile_context("linkedlist_insert"):
            start_time = time.time()
            for i in range(min(1000, size // 10)):
                linked_list.insert(random.randint(0, 1000))
            insert_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Insert", "Linked List", size, insert_time,
            self.profiler_results["linkedlist_insert"]["memory_delta"],
            self.profiler_results["linkedlist_insert"]["cpu_usage"],
            "O(1)"
        ))
        
        # Search operation
        with self.profile_context("linkedlist_search"):
            start_time = time.time()
            for i in range(min(1000, size // 10)):
                target = random.randint(0, size)
                linked_list.search(target)
            search_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Search", "Linked List", size, search_time,
            self.profiler_results["linkedlist_search"]["memory_delta"],
            self.profiler_results["linkedlist_search"]["cpu_usage"],
            "O(n)"
        ))
        
        # Delete operation
        with self.profile_context("linkedlist_delete"):
            start_time = time.time()
            for i in range(min(1000, size // 10)):
                target = random.randint(0, size)
                linked_list.delete(target)
            delete_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Delete", "Linked List", size, delete_time,
            self.profiler_results["linkedlist_delete"]["memory_delta"],
            self.profiler_results["linkedlist_delete"]["cpu_usage"],
            "O(n)"
        ))
        
        return results

    def benchmark_hashmap_operations(self, size: int) -> List[PerformanceResult]:
        """Benchmark hashmap operations"""
        results = []
        hashmap = {i: f"value_{i}" for i in range(size)}
        
        # Insert operation
        with self.profile_context("hashmap_insert"):
            start_time = time.time()
            for i in range(min(1000, size // 10)):
                hashmap[f"key_{i}"] = f"value_{i}"
            insert_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Insert", "HashMap", size, insert_time,
            self.profiler_results["hashmap_insert"]["memory_delta"],
            self.profiler_results["hashmap_insert"]["cpu_usage"],
            "O(1)"
        ))
        
        # Search operation
        with self.profile_context("hashmap_search"):
            start_time = time.time()
            for i in range(min(1000, size // 10)):
                target = random.randint(0, size)
                _ = target in hashmap
            search_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Search", "HashMap", size, search_time,
            self.profiler_results["hashmap_search"]["memory_delta"],
            self.profiler_results["hashmap_search"]["cpu_usage"],
            "O(1)"
        ))
        
        # Delete operation
        with self.profile_context("hashmap_delete"):
            start_time = time.time()
            temp_hashmap = hashmap.copy()
            for i in range(min(1000, size // 10)):
                target = random.randint(0, size)
                temp_hashmap.pop(target, None)
            delete_time = time.time() - start_time
            
        results.append(PerformanceResult(
            "Delete", "HashMap", size, delete_time,
            self.profiler_results["hashmap_delete"]["memory_delta"],
            self.profiler_results["hashmap_delete"]["cpu_usage"],
            "O(1)"
        ))
        
        return results

    def run_comprehensive_benchmark(self, sizes: List[int]) -> pd.DataFrame:
        """Run comprehensive benchmark across all data structures"""
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_operations = len(sizes) * 3  # 3 data structures
        current_operation = 0
        
        for size in sizes:
            # Array benchmark
            status_text.text(f"Benchmarking Array operations (size: {size:,})")
            array_results = self.benchmark_array_operations(size)
            all_results.extend(array_results)
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
            
            # Linked List benchmark
            status_text.text(f"Benchmarking Linked List operations (size: {size:,})")
            linkedlist_results = self.benchmark_linked_list_operations(size)
            all_results.extend(linkedlist_results)
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
            
            # HashMap benchmark
            status_text.text(f"Benchmarking HashMap operations (size: {size:,})")
            hashmap_results = self.benchmark_hashmap_operations(size)
            all_results.extend(hashmap_results)
            current_operation += 1
            progress_bar.progress(current_operation / total_operations)
            
            # Clean up memory
            gc.collect()
        
        status_text.text("Benchmark completed!")
        progress_bar.empty()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Operation': result.operation,
                'Data Structure': result.data_structure,
                'Size': result.size,
                'Time (seconds)': result.time_taken,
                'Memory Usage (MB)': result.memory_usage,
                'CPU Usage (%)': result.cpu_usage,
                'Complexity': result.complexity
            }
            for result in all_results
        ])
        
        return df

def create_performance_visualizations(df: pd.DataFrame):
    """Create comprehensive performance visualizations"""
    
    # Time complexity comparison
    fig_time = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Insert Operations', 'Search Operations', 'Delete Operations', 'Overall Performance'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"type": "scatter3d"}]]
    )
    
    operations = ['Insert', 'Search', 'Delete']
    colors = {'Array': '#ff7f0e', 'Linked List': '#2ca02c', 'HashMap': '#1f77b4'}
    
    for i, operation in enumerate(operations):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        op_data = df[df['Operation'] == operation]
        
        for ds in op_data['Data Structure'].unique():
            ds_data = op_data[op_data['Data Structure'] == ds]
            
            fig_time.add_trace(
                go.Scatter(
                    x=ds_data['Size'],
                    y=ds_data['Time (seconds)'],
                    mode='lines+markers',
                    name=f'{ds} - {operation}',
                    line=dict(color=colors[ds]),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
    
    # Overall 3D performance
    fig_time.add_trace(
        go.Scatter3d(
            x=df['Size'],
            y=df['Time (seconds)'],
            z=df['Memory Usage (MB)'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['CPU Usage (%)'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="CPU Usage (%)")
            ),
            text=df['Data Structure'] + ' - ' + df['Operation'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Size: %{x}<br>' +
                         'Time: %{y:.6f}s<br>' +
                         'Memory: %{z:.2f}MB<br>' +
                         'CPU: %{marker.color:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig_time.update_layout(
        title_text="Comprehensive Performance Analysis",
        height=800,
        showlegend=True
    )
    
    return fig_time

def create_complexity_comparison():
    """Create theoretical vs actual complexity comparison"""
    complexity_data = {
        'Operation': ['Insert', 'Search', 'Delete'] * 3,
        'Data Structure': ['Array'] * 3 + ['Linked List'] * 3 + ['HashMap'] * 3,
        'Theoretical Complexity': ['O(n)', 'O(n)', 'O(n)', 'O(1)', 'O(n)', 'O(n)', 'O(1)', 'O(1)', 'O(1)'],
        'Complexity Score': [3, 3, 3, 1, 3, 3, 1, 1, 1]  # Higher is worse
    }
    
    complexity_df = pd.DataFrame(complexity_data)
    
    fig = px.bar(
        complexity_df,
        x='Operation',
        y='Complexity Score',
        color='Data Structure',
        barmode='group',
        title='Theoretical Time Complexity Comparison',
        labels={'Complexity Score': 'Complexity Score (Lower is Better)'},
        color_discrete_map={
            'Array': '#ff7f0e',
            'Linked List': '#2ca02c',
            'HashMap': '#1f77b4'
        }
    )
    
    # Add complexity annotations
    annotations = []
    for i, row in complexity_df.iterrows():
        annotations.append(
            dict(
                x=row['Operation'],
                y=row['Complexity Score'] + 0.1,
                text=row['Theoretical Complexity'],
                showarrow=False,
                font=dict(size=10)
            )
        )
    
    fig.update_layout(annotations=annotations)
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Structures Performance Analyzer</h1>
        <p>Comprehensive performance comparison of Arrays, Linked Lists, and HashMaps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Benchmark Configuration")
    
    # Size selection
    size_option = st.sidebar.selectbox(
        "Select data sizes to benchmark:",
        ["Quick Test", "Standard Test", "Comprehensive Test", "Custom"]
    )
    
    if size_option == "Quick Test":
        sizes = [100, 500, 1000]
    elif size_option == "Standard Test":
        sizes = [1000, 5000, 10000, 20000]
    elif size_option == "Comprehensive Test":
        sizes = [1000, 5000, 10000, 20000, 50000, 100000]
    else:  # Custom
        custom_sizes = st.sidebar.text_input(
            "Enter sizes (comma-separated):",
            "1000,5000,10000"
        )
        try:
            sizes = [int(x.strip()) for x in custom_sizes.split(",")]
        except:
            sizes = [1000, 5000, 10000]
    
    # Performance options
    st.sidebar.header("🎯 Analysis Options")
    show_profiler = st.sidebar.checkbox("Show detailed profiler output", value=True)
    show_memory = st.sidebar.checkbox("Include memory analysis", value=True)
    show_cpu = st.sidebar.checkbox("Include CPU usage analysis", value=True)
    
    # Run benchmark button
    if st.sidebar.button("🚀 Run Benchmark", type="primary"):
        analyzer = DataStructureAnalyzer()
        
        with st.spinner("Running comprehensive benchmark..."):
            df = analyzer.run_comprehensive_benchmark(sizes)
            
        st.session_state['benchmark_results'] = df
        st.session_state['profiler_results'] = analyzer.profiler_results
        st.success("Benchmark completed successfully!")
    
    # Display results if available
    if 'benchmark_results' in st.session_state:
        df = st.session_state['benchmark_results']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance Overview",
            "📊 Detailed Analysis",
            "🧠 Memory & CPU",
            "⚡ Profiler Results",
            "📋 Raw Data"
        ])
        
        with tab1:
            st.header("Performance Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>Fastest Insert</h3>
                    <p><strong>HashMap</strong></p>
                    <p>O(1) complexity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Fastest Search</h3>
                    <p><strong>HashMap</strong></p>
                    <p>O(1) complexity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>Memory Efficient</h3>
                    <p><strong>Array</strong></p>
                    <p>Contiguous memory</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>Most Versatile</h3>
                    <p><strong>HashMap</strong></p>
                    <p>Key-value pairs</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance visualization
            fig_time = create_performance_visualizations(df)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Complexity comparison
            fig_complexity = create_complexity_comparison()
            st.plotly_chart(fig_complexity, use_container_width=True)
        
        with tab2:
            st.header("Detailed Performance Analysis")
            
            # Operation selection
            selected_operation = st.selectbox(
                "Select operation to analyze:",
                df['Operation'].unique()
            )
            
            op_data = df[df['Operation'] == selected_operation]
            
            # Time performance chart
            fig_detailed = px.line(
                op_data,
                x='Size',
                y='Time (seconds)',
                color='Data Structure',
                title=f'{selected_operation} Operation - Time Performance',
                markers=True,
                log_y=True
            )
            st.plotly_chart(fig_detailed, use_container_width=True)
            
            # Performance recommendations
            st.subheader("Performance Recommendations")
            
            if selected_operation == "Insert":
                st.markdown("""
                <div class="performance-good">
                    <h4>✅ Best Choice: HashMap</h4>
                    <p>HashMap provides O(1) average time complexity for insertions, making it ideal for scenarios with frequent insertions.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="performance-warning">
                    <h4>⚠️ Consider: Linked List</h4>
                    <p>Linked List also provides O(1) insertion at the head, but may have higher memory overhead due to pointer storage.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif selected_operation == "Search":
                st.markdown("""
                <div class="performance-good">
                    <h4>✅ Best Choice: HashMap</h4>
                    <p>HashMap provides O(1) average time complexity for searches, significantly outperforming linear search structures.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            if show_memory and show_cpu:
                st.header("Memory & CPU Analysis")
                
                # Memory usage chart
                fig_memory = px.bar(
                    df,
                    x='Data Structure',
                    y='Memory Usage (MB)',
                    color='Operation',
                    title='Memory Usage by Data Structure and Operation',
                    barmode='group'
                )
                st.plotly_chart(fig_memory, use_container_width=True)
                
                # CPU usage chart
                fig_cpu = px.scatter(
                    df,
                    x='Time (seconds)',
                    y='CPU Usage (%)',
                    color='Data Structure',
                    size='Size',
                    title='CPU Usage vs Execution Time',
                    hover_data=['Operation']
                )
                st.plotly_chart(fig_cpu, use_container_width=True)
                
                # Resource efficiency analysis
                st.subheader("Resource Efficiency Analysis")
                
                # Calculate efficiency score
                df['Efficiency Score'] = 1 / (df['Time (seconds)'] * df['Memory Usage (MB)'] * df['CPU Usage (%)'])
                
                efficiency_summary = df.groupby('Data Structure')['Efficiency Score'].mean().reset_index()
                efficiency_summary = efficiency_summary.sort_values('Efficiency Score', ascending=False)
                
                fig_efficiency = px.bar(
                    efficiency_summary,
                    x='Data Structure',
                    y='Efficiency Score',
                    title='Overall Resource Efficiency Score (Higher is Better)',
                    color='Efficiency Score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
        
        with tab4:
            if show_profiler and 'profiler_results' in st.session_state:
                st.header("Detailed Profiler Results")
                
                profiler_results = st.session_state['profiler_results']
                
                # Show profiler data for each operation
                for operation, data in profiler_results.items():
                    with st.expander(f"📊 {operation.replace('_', ' ').title()} Profile"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Execution Time", f"{data['execution_time']:.6f}s")
                            st.metric("Memory Delta", f"{data['memory_delta']:.2f}MB")
                        
                        with col2:
                            st.metric("CPU Usage", f"{data['cpu_usage']:.1f}%")
                        
                        st.text("Detailed Profile:")
                        st.code(data['profile_data'], language='text')
        
        with tab5:
            st.header("Raw Benchmark Data")
            
            # Display raw data
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="benchmark_results.csv",
                mime="text/csv"
            )
    
    else:
        # Show introduction if no benchmark has been run
        st.markdown("""
        ## Welcome to the Data Structures Performance Analyzer!
        
        This tool provides comprehensive performance analysis of three fundamental data structures:
        
        ### 🔍 What We Analyze
        - **Arrays (Lists)**: Contiguous memory allocation, great for random access
        - **Linked Lists**: Dynamic memory allocation, efficient insertion/deletion
        - **HashMaps (Dictionaries)**: Key-value pairs, excellent for lookups
        
        ### 📊 Operations Tested
        - **Insert**: Adding new elements
        - **Search**: Finding specific elements
        - **Delete**: Removing elements
        
        ### 🛠️ Profiling Tools Used
        - **cProfile**: Function-level profiling
        - **memory_profiler**: Memory usage tracking
        - **psutil**: System resource monitoring
        - **Custom timing**: Precise execution time measurement
        
        ### 🚀 Getting Started
        1. Configure your benchmark settings in the sidebar
        2. Choose data sizes to test
        3. Click "Run Benchmark" to start analysis
        4. Explore results in the interactive tabs
        
        **Note**: Larger datasets will take longer to process but provide more accurate performance insights.
        """)

if __name__ == "__main__":
    main()