#!/usr/bin/env python3
"""
Data Analysis and Visualization for Drowsiness Detection
Advanced analytics and reporting capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DrowsinessDataAnalyzer:
    """
    Advanced data analyzer for drowsiness detection results.
    """
    
    def __init__(self, data_file=None):
        """
        Initialize data analyzer.
        
        Args:
            data_file: Path to data file (CSV or JSON)
        """
        self.data = None
        self.df = None
        
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, file_path):
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
        """
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    self.data = json.load(f)
                self.df = pd.DataFrame(self.data)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"Data loaded: {len(self.df)} records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def analyze_drowsiness_patterns(self):
        """
        Analyze drowsiness patterns in the data.
        
        Returns:
            dict: Analysis results
        """
        if self.df is None:
            return {}
        
        analysis = {}
        
        # Basic statistics
        analysis['total_detections'] = len(self.df)
        analysis['drowsiness_detections'] = len(self.df[self.df['drowsiness_detected'] == True])
        analysis['drowsiness_rate'] = analysis['drowsiness_detections'] / analysis['total_detections']
        
        # Severity analysis
        if 'drowsiness_severity' in self.df.columns:
            severity_counts = self.df['drowsiness_severity'].value_counts()
            analysis['severity_distribution'] = severity_counts.to_dict()
        
        # Temporal analysis
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            analysis['time_span'] = {
                'start': self.df['timestamp'].min(),
                'end': self.df['timestamp'].max(),
                'duration': self.df['timestamp'].max() - self.df['timestamp'].min()
            }
        
        # EAR analysis
        if 'ear' in self.df.columns:
            analysis['ear_statistics'] = {
                'mean': self.df['ear'].mean(),
                'std': self.df['ear'].std(),
                'min': self.df['ear'].min(),
                'max': self.df['ear'].max(),
                'median': self.df['ear'].median()
            }
        
        return analysis
    
    def calculate_perclos_metrics(self, time_window=60):
        """
        Calculate PERCLOS metrics over time windows.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            pandas.Series: PERCLOS values over time
        """
        if self.df is None or 'ear' not in self.df.columns:
            return pd.Series()
        
        # Group by time windows
        self.df['time_window'] = pd.to_datetime(self.df['timestamp']).dt.floor(f'{time_window}S')
        
        perclos_values = []
        for window, group in self.df.groupby('time_window'):
            ear_values = group['ear'].values
            perclos = (ear_values < 0.25).sum() / len(ear_values) * 100
            perclos_values.append(perclos)
        
        return pd.Series(perclos_values, index=self.df['time_window'].unique())
    
    def detect_episodes(self, min_duration=5, threshold=0.25):
        """
        Detect drowsiness episodes.
        
        Args:
            min_duration: Minimum duration in frames
            threshold: EAR threshold
            
        Returns:
            list: Drowsiness episodes
        """
        if self.df is None or 'ear' not in self.df.columns:
            return []
        
        episodes = []
        current_episode = None
        
        for idx, row in self.df.iterrows():
            if row['ear'] < threshold:
                if current_episode is None:
                    current_episode = {'start': idx, 'start_time': row['timestamp']}
            else:
                if current_episode is not None:
                    current_episode['end'] = idx
                    current_episode['end_time'] = row['timestamp']
                    current_episode['duration'] = idx - current_episode['start']
                    
                    if current_episode['duration'] >= min_duration:
                        episodes.append(current_episode)
                    
                    current_episode = None
        
        return episodes
    
    def generate_report(self, output_file=None):
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Output file path
            
        Returns:
            str: Report content
        """
        analysis = self.analyze_drowsiness_patterns()
        perclos_metrics = self.calculate_perclos_metrics()
        episodes = self.detect_episodes()
        
        report = []
        report.append("="*60)
        report.append("DROWSINESS DETECTION ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"  Total detections: {analysis.get('total_detections', 0)}")
        report.append(f"  Drowsiness detections: {analysis.get('drowsiness_detections', 0)}")
        report.append(f"  Drowsiness rate: {analysis.get('drowsiness_rate', 0):.2%}")
        report.append("")
        
        # Severity analysis
        if 'severity_distribution' in analysis:
            report.append("SEVERITY DISTRIBUTION:")
            for severity, count in analysis['severity_distribution'].items():
                report.append(f"  {severity}: {count}")
            report.append("")
        
        # EAR statistics
        if 'ear_statistics' in analysis:
            ear_stats = analysis['ear_statistics']
            report.append("EAR STATISTICS:")
            report.append(f"  Mean: {ear_stats['mean']:.3f}")
            report.append(f"  Std: {ear_stats['std']:.3f}")
            report.append(f"  Min: {ear_stats['min']:.3f}")
            report.append(f"  Max: {ear_stats['max']:.3f}")
            report.append("")
        
        # Episodes
        report.append(f"DROWSINESS EPISODES: {len(episodes)}")
        for i, episode in enumerate(episodes[:10]):  # Show first 10
            duration = episode['duration']
            report.append(f"  Episode {i+1}: {duration} frames")
        if len(episodes) > 10:
            report.append(f"  ... and {len(episodes) - 10} more episodes")
        report.append("")
        
        # PERCLOS metrics
        if not perclos_metrics.empty:
            report.append("PERCLOS METRICS:")
            report.append(f"  Average PERCLOS: {perclos_metrics.mean():.2f}%")
            report.append(f"  Max PERCLOS: {perclos_metrics.max():.2f}%")
            report.append(f"  Min PERCLOS: {perclos_metrics.min():.2f}%")
            report.append("")
        
        report.append("="*60)
        
        report_content = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"Report saved to: {output_file}")
        
        return report_content
    
    def plot_ear_timeline(self, output_file=None):
        """
        Plot EAR values over time.
        
        Args:
            output_file: Output file path for plot
        """
        if self.df is None or 'ear' not in self.df.columns:
            print("No EAR data available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in self.df.columns:
            timestamps = pd.to_datetime(self.df['timestamp'])
        else:
            timestamps = range(len(self.df))
        
        plt.plot(timestamps, self.df['ear'], 'b-', alpha=0.7, label='EAR')
        plt.axhline(y=0.25, color='r', linestyle='--', label='Threshold')
        
        # Highlight drowsiness episodes
        episodes = self.detect_episodes()
        for episode in episodes:
            start_idx = episode['start']
            end_idx = episode['end']
            plt.axvspan(timestamps.iloc[start_idx], timestamps.iloc[end_idx], 
                       alpha=0.3, color='red')
        
        plt.xlabel('Time')
        plt.ylabel('Eye Aspect Ratio (EAR)')
        plt.title('EAR Timeline with Drowsiness Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
        
        plt.show()
    
    def plot_severity_distribution(self, output_file=None):
        """
        Plot drowsiness severity distribution.
        
        Args:
            output_file: Output file path for plot
        """
        if self.df is None or 'drowsiness_severity' not in self.df.columns:
            print("No severity data available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        
        severity_counts = self.df['drowsiness_severity'].value_counts()
        colors = ['green', 'orange', 'red', 'gray']
        
        plt.pie(severity_counts.values, labels=severity_counts.index, 
               colors=colors[:len(severity_counts)], autopct='%1.1f%%')
        plt.title('Drowsiness Severity Distribution')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")
        
        plt.show()
    
    def export_summary(self, output_file):
        """
        Export summary statistics to JSON.
        
        Args:
            output_file: Output file path
        """
        analysis = self.analyze_drowsiness_patterns()
        perclos_metrics = self.calculate_perclos_metrics()
        episodes = self.detect_episodes()
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_statistics': analysis,
            'perclos_metrics': {
                'mean': float(perclos_metrics.mean()) if not perclos_metrics.empty else 0,
                'max': float(perclos_metrics.max()) if not perclos_metrics.empty else 0,
                'min': float(perclos_metrics.min()) if not perclos_metrics.empty else 0
            },
            'episodes_count': len(episodes),
            'episodes_summary': [
                {
                    'start_frame': ep['start'],
                    'end_frame': ep['end'],
                    'duration_frames': ep['duration'],
                    'start_time': str(ep['start_time']),
                    'end_time': str(ep['end_time'])
                }
                for ep in episodes[:20]  # Limit to first 20 episodes
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary exported to: {output_file}")

class RealTimeAnalyzer:
    """
    Real-time data analyzer for live drowsiness detection.
    """
    
    def __init__(self, window_size=300):
        """
        Initialize real-time analyzer.
        
        Args:
            window_size: Size of sliding window (frames)
        """
        self.window_size = window_size
        self.data_buffer = []
        self.alerts = []
    
    def add_data_point(self, data_point):
        """
        Add new data point to analysis buffer.
        
        Args:
            data_point: Dictionary containing detection data
        """
        self.data_buffer.append(data_point)
        
        # Keep only recent data
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
    
    def get_current_metrics(self):
        """
        Get current metrics from recent data.
        
        Returns:
            dict: Current metrics
        """
        if not self.data_buffer:
            return {}
        
        recent_data = self.data_buffer[-100:]  # Last 100 frames
        
        metrics = {
            'total_frames': len(recent_data),
            'drowsiness_detections': sum(1 for d in recent_data if d.get('drowsiness_detected', False)),
            'avg_ear': np.mean([d.get('ear', 0) for d in recent_data]),
            'ear_variance': np.var([d.get('ear', 0) for d in recent_data]),
            'alert_frequency': len([a for a in self.alerts if a['time'] > time.time() - 300]) / 5  # alerts per minute
        }
        
        metrics['drowsiness_rate'] = metrics['drowsiness_detections'] / metrics['total_frames']
        
        return metrics
    
    def should_alert(self, current_data):
        """
        Determine if alert should be triggered based on current data.
        
        Args:
            current_data: Current detection data
            
        Returns:
            bool: Whether to trigger alert
        """
        metrics = self.get_current_metrics()
        
        # Alert conditions
        if metrics['drowsiness_rate'] > 0.3:  # 30% drowsiness rate
            return True
        
        if metrics['avg_ear'] < 0.2:  # Very low EAR
            return True
        
        if metrics['alert_frequency'] < 2:  # Not too many recent alerts
            if current_data.get('drowsiness_severity') == 'high':
                return True
        
        return False
    
    def add_alert(self, alert_data):
        """
        Add alert to history.
        
        Args:
            alert_data: Alert information
        """
        self.alerts.append({
            'time': time.time(),
            'data': alert_data
        })
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts.pop(0)

# Utility functions
def analyze_session_data(session_file):
    """
    Analyze data from a detection session.
    
    Args:
        session_file: Path to session data file
        
    Returns:
        dict: Analysis results
    """
    analyzer = DrowsinessDataAnalyzer(session_file)
    return analyzer.analyze_drowsiness_patterns()

def generate_session_report(session_file, output_dir='reports'):
    """
    Generate comprehensive report for a session.
    
    Args:
        session_file: Path to session data file
        output_dir: Output directory for reports
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DrowsinessDataAnalyzer(session_file)
    
    # Generate report
    report_file = os.path.join(output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    analyzer.generate_report(report_file)
    
    # Generate plots
    plot_file = os.path.join(output_dir, f"ear_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    analyzer.plot_ear_timeline(plot_file)
    
    # Export summary
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    analyzer.export_summary(summary_file)
    
    print(f"Session analysis completed. Files saved to: {output_dir}") 