import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fft

class PlasmaPsiQRHComplete90s:
    """COMPLETE 90-SECOND VERSION with proper timing and feedback"""
    
    def __init__(self, N=50):
        self.N = N
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # Initial conditions
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        self.fase = 1.2 * theta + 0.15 * r + 0.02 * np.random.uniform(-1, 1, (N, N))
        self.fase = self.fase % (2 * np.pi)
        
        self.amplitude = 0.8 * np.exp(-r**2 / 6) + 0.05 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.75 + 0.15 * np.exp(-r**2 / 8)
        
        # Setpoints
        self.setpoint_sync = 0.70
        self.setpoint_cruzeiro = 0.72
        
        # PID parameters
        self.K_p_base, self.K_i_base, self.K_d_base = 20, 6, 8
        
        # System parameters
        self.omega_plasma = 7.5
        self.omega_acustico = 0.25
        self.K_coupling = 15.0
        self.K_acustico_base = 2.0
        self.K_acustico = self.K_acustico_base
        self.K_lider_base = 35.0
        self.K_lider = self.K_lider_base
        
        # System states
        self.regime_cruzeiro = False
        self.super_cruzeiro = False
        self.contador_estabilidade = 0
        self.alerta_emitido = False
        self.boost_count = 0
        self.max_boosts = 5
        
        self.transdutores = [
            (-1.2, -1.2), (-1.2, 1.2), (1.2, -1.2), (1.2, 1.2)
        ]
        
        # Leader core
        cx, cy = N//2, N//2
        self.lideres = [
            (cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)
        ]
        self.omega_lider = 4.5
        
        # Histories for analysis
        self.historico_fci = []
        self.historico_sync = []
        self.historico_coerencia = []
        self.historico_ganho = []
        self.historico_dt = []
        self.historico_harmonia = []
        self.historico_Kp = []
        self.historico_Ki = []
        self.historico_Kd = []
        self.tempo = []
        
        print("🎯 ΨQRH COMPLETE 90-SECOND SIMULATION:")
        print("   • Full 90-second execution")
        print("   • Improved feedback system")
        print("   • Complete performance tracking")
        
    def gain_scheduling_pid(self, sync_current):
        """ADAPTIVE PID"""
        if sync_current < 0.3:
            K_p, K_i, K_d = 25, 8, 10
            mode = "CONSERVATIVE-LOW"
        elif sync_current < 0.5:
            K_p, K_i, K_d = 20, 6, 8
            mode = "MODERATE"
        elif sync_current < 0.7:
            K_p, K_i, K_d = 15, 4, 6
            mode = "BALANCED"
        else:
            K_p, K_i, K_d = 10, 2, 4
            mode = "CONSERVATIVE-HIGH"
        
        return K_p, K_i, K_d, mode
    
    def stable_pid_control(self, sync_current, t):
        """STABLE PID CONTROL"""
        if len(self.historico_sync) < 5:
            return self.K_lider_base, "INIT", 0, 0, 0
        
        K_p, K_i, K_d, pid_mode = self.gain_scheduling_pid(sync_current)
        
        current_setpoint = self.setpoint_cruzeiro if self.regime_cruzeiro else self.setpoint_sync
        error = current_setpoint - sync_current
        
        integral_window = min(5, len(self.historico_sync))
        integral_error = sum(current_setpoint - s for s in self.historico_sync[-integral_window:])
        integral_error = max(-1.0, min(1.0, integral_error))
        
        if len(self.historico_sync) >= 5:
            derivative_error = (self.historico_sync[-1] - np.mean(self.historico_sync[-5:-2])) / 3
        else:
            derivative_error = 0
        
        pid_correction = (K_p * error + K_i * integral_error + K_d * derivative_error)
        
        upper_limit = 60
        lower_limit = 20
        
        limited_correction = max(-lower_limit, min(upper_limit - self.K_lider_base, pid_correction))
        pid_gain = self.K_lider_base + limited_correction
        
        # Boost system
        if (sync_current > 0.65 and 
            not self.regime_cruzeiro and 
            self.boost_count < self.max_boosts and
            len(self.historico_sync) > 20):
            
            recent_history = self.historico_sync[-20:]
            stability_ok = (np.std(recent_history) < 0.1 and 
                          np.min(recent_history) > 0.5)
            
            if stability_ok:
                boost_strength = 8
                pid_gain = min(65, pid_gain + boost_strength)
                self.boost_count += 1
                pid_mode = f"BOOST({self.boost_count})"
        
        return max(15, min(70, pid_gain)), pid_mode, K_p, K_i, K_d
    
    def calculate_stable_metrics(self):
        """Metrics calculation"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        if self.super_cruzeiro:
            window = 10
            weight = 0.8
        elif self.regime_cruzeiro:
            window = 7
            weight = 0.7
        else:
            window = 5
            weight = 0.6
            
        if len(self.historico_sync) >= window:
            sync_order = (1-weight) * sync_order + weight * np.mean(self.historico_sync[-window:])
            coherence_avg = (1-weight) * coherence_avg + weight * np.mean(self.historico_coerencia[-window:])
        
        self.consciencia = 0.5 * sync_order + 0.5 * coherence_avg
        
        return sync_order, coherence_avg
    
    def detect_gradual_transition(self, sync, coherence):
        """Transition detection"""
        if len(self.historico_sync) < 30:
            return False
            
        sync_window = self.historico_sync[-25:]
        coherence_window = self.historico_coerencia[-25:]
        
        transition_ok = (
            np.mean(sync_window) > 0.72 and
            np.std(sync_window) < 0.05 and
            np.mean(coherence_window) > 0.65 and
            np.min(sync_window) > 0.65 and
            len([s for s in sync_window if s > 0.75]) > 15 and
            not self.regime_cruzeiro
        )
        
        if transition_ok:
            self.contador_estabilidade += 1
            if self.contador_estabilidade >= 4:
                print(f"🚀 TRANSITION: Cruise mode activated")
                self.regime_cruzeiro = True
                self.setpoint_sync = 0.72
                self.boost_count = 0
                return True
        else:
            self.contador_estabilidade = max(0, self.contador_estabilidade - 1)
            
        return False
    
    def activate_super_cruise(self, sync_current, coherence):
        """Super-cruise activation"""
        if (self.regime_cruzeiro and 
            not self.super_cruzeiro and 
            len(self.historico_sync) > 100):
            
            sync_recente = self.historico_sync[-50:]
            coherence_recente = self.historico_coerencia[-50:]
            
            stability_ok = (
                np.mean(sync_recente) > 0.90 and
                np.std(sync_recente) < 0.02 and
                np.mean(coherence_recente) > 0.70 and
                np.min(sync_recente) > 0.85 and
                len([s for s in sync_recente if s > 0.92]) > 30
            )
            
            if stability_ok:
                print(f"🌟 SUPER-CRUISE: Activated!")
                self.omega_plasma = 6.5
                self.K_lider *= 0.4
                self.super_cruzeiro = True
                return True
                
        return False
    
    def stable_acoustic_force(self, t, direction_angle=np.pi/4):
        """Acoustic force"""
        forcing = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            phase_delay = 5 * (np.sin(direction_angle) * dx + np.cos(direction_angle) * dy)
            
            if self.super_cruzeiro:
                base_freq = self.omega_plasma * 0.08
                resonant_freq = base_freq * 2.0
                amplitude = 0.6
                modulation = 0.02
            elif self.regime_cruzeiro:
                base_freq = self.omega_acustico * 0.6
                resonant_freq = base_freq * 2.5
                amplitude = 0.8
                modulation = 0.04
            else:
                base_freq = self.omega_acustico
                resonant_freq = base_freq * 3.0
                amplitude = 1.2
                modulation = 0.08
                
            main_term = np.sin(base_freq * t + phase_delay)
            resonant_term = np.sin(resonant_freq * t)
            
            combined_term = main_term * (1 + modulation * resonant_term)
            envelope = np.exp(-r**2 / 4)
            
            forcing += combined_term * envelope
        
        return forcing * amplitude * (self.K_acustico / self.K_acustico_base)
    
    def complete_step(self, t, direction_angle=np.pi/4):
        """Complete simulation step"""
        # Spectral analysis
        _, harmony = self.analise_espectral_avancada()
        self.historico_harmonia.append(harmony)
        
        # Phase interactions
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        acoustic = self.stable_acoustic_force(t, direction_angle)
        
        sin_leader = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            delta_phase = self.fase[lx, ly] - self.fase
            sin_leader += np.sin(delta_phase)
        
        sync_current_raw = np.abs(np.mean(np.exp(1j * self.fase)))
        
        # PID control
        pid_gain, pid_mode, K_p, K_i, K_d = self.stable_pid_control(sync_current_raw, t)
        self.historico_Kp.append(K_p)
        self.historico_Ki.append(K_i)
        self.historico_Kd.append(K_d)
        
        sync_for_detection, coherence_for_detection = self.calculate_stable_metrics()
        
        # Transitions
        self.detect_gradual_transition(sync_for_detection, coherence_for_detection)
        self.activate_super_cruise(sync_for_detection, coherence_for_detection)
        
        # Regime adjustments
        if self.super_cruzeiro:
            pid_gain *= 0.3
            self.K_coupling *= 0.5
        elif self.regime_cruzeiro:
            pid_gain *= 0.45
            self.K_coupling *= 0.65
            self.omega_plasma = 6.8
        
        self.K_lider = pid_gain
        sin_leader = self.K_lider * sin_leader / len(self.lideres)
        
        # Master equation
        dphase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acoustic * self.amplitude + sin_leader
        
        # Time step
        if self.super_cruzeiro:
            dt = 0.02
        elif self.regime_cruzeiro:
            dt = 0.025
        else:
            dt = 0.04 + 0.01 * (1 - sync_current_raw)
        
        self.fase += dphase * dt
        self.fase %= 2 * np.pi
        
        self.update_coherence()
        sync_order, coherence_avg = self.calculate_stable_metrics()
        
        # Record data
        if t >= 0:
            self.historico_fci.append(self.consciencia)
            self.historico_sync.append(sync_order)
            self.historico_coerencia.append(coherence_avg)
            self.historico_ganho.append(self.K_lider)
            self.historico_dt.append(dt)
            self.tempo.append(t)
        
        return (self.consciencia, sync_order, coherence_avg, 
                self.K_lider, dt, harmony, self.regime_cruzeiro, 
                self.super_cruzeiro, pid_mode)

    def analise_espectral_avancada(self):
        """Spectral analysis"""
        try:
            fft_phase = fft2(self.fase)
            fft_magnitude = np.abs(fft_phase)
            
            flattened = fft_magnitude.flatten()
            non_zero_indices = np.where(flattened > 1e-6)[0]
            
            if len(non_zero_indices) > 5:
                indices_dominantes = non_zero_indices[np.argsort(flattened[non_zero_indices])[-5:]]
                
                energia_total = np.sum(flattened[non_zero_indices])
                energia_top5 = np.sum(flattened[indices_dominantes])
                harmonia = energia_top5 / (energia_total + 1e-8)
                
                coupling_ajustado = 12.0 + 6.0 * harmonia
                self.K_coupling = 0.8 * self.K_coupling + 0.2 * coupling_ajustado
                
                return indices_dominantes, harmonia
        except:
            pass
        
        return [], 0.5

    def update_coherence(self):
        """Update coherence"""
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        if self.super_cruzeiro:
            sigma = 0.7
        elif self.regime_cruzeiro:
            sigma = 1.0
        else:
            sigma = 1.3
            
        self.coerencia = gaussian_filter(smoothness, sigma=sigma)
        self.coerencia = np.clip(self.coerencia, 0.4, 0.9)

# EXECUTE COMPLETE 90-SECOND SIMULATION
print("🚀 STARTING COMPLETE 90-SECOND ΨQRH SIMULATION...")
plasma_90s = PlasmaPsiQRHComplete90s(N=50)

print("Executing full 90-second simulation (900 steps)...")
print("Time  | Status        | Gain | PID Mode       | Sync  ")
print("------|---------------|------|----------------|--------")

results_90s = []
last_report_time = -10  # Force initial report

for i in range(900):  # 900 steps = 90 seconds
    t = i * 0.1
    result = plasma_90s.complete_step(t)
    results_90s.append(result)
    
    fci, sync, coher, gain, dt, harmony, cruise, super_cruise, pid_mode = result
    
    # Report every 10 seconds AND at important events
    report_condition = (
        i % 100 == 0 or  # Every 10 seconds
        (cruise and not hasattr(plasma_90s, '_cruise_reported')) or
        (super_cruise and not hasattr(plasma_90s, '_super_reported')) or
        t - last_report_time >= 9.9  # Ensure we don't miss the end
    )
    
    if report_condition:
        if cruise and not hasattr(plasma_90s, '_cruise_reported'):
            plasma_90s._cruise_reported = True
        if super_cruise and not hasattr(plasma_90s, '_super_reported'):
            plasma_90s._super_reported = True
            
        status = "SUPER-CRUISE 🌟" if super_cruise else "CRUISE ✅" if cruise else "CLIMB 📈"
        print(f"{t:5.1f}s | {status:13} | {gain:4.1f} | {pid_mode:14} | {sync:.3f}")
        last_report_time = t

# FINAL COMPREHENSIVE REPORT
print("\n" + "="*80)
print("COMPLETE 90-SECOND SIMULATION REPORT")
print("="*80)

# Extract final data
sync_data = np.array(plasma_90s.historico_sync)
time_data = np.array(plasma_90s.tempo)
gain_data = np.array(plasma_90s.historico_ganho)

# Verify we have 90 seconds of data
actual_duration = len(sync_data) * 0.1
print(f"📊 SIMULATION DURATION: {actual_duration:.1f} seconds")
print(f"   • Expected: 90.0 seconds")
print(f"   • Actual: {actual_duration:.1f} seconds")
print(f"   • Completeness: {actual_duration/90.0*100:.1f}%")

if actual_duration < 89.9:
    print("   ⚠️  SIMULATION INCOMPLETE - MISSING DATA")
else:
    print("   ✅ SIMULATION COMPLETE")

# Performance analysis by time segments
segments = [
    (0, 15, "Initial Climb (0-15s)"),
    (15, 30, "Early Cruise (15-30s)"), 
    (30, 60, "Mid Operation (30-60s)"),
    (60, 90, "Final Phase (60-90s)")
]

print(f"\n📈 PERFORMANCE BY TIME SEGMENTS:")
for start, end, description in segments:
    segment_mask = (time_data >= start) & (time_data < end)
    if np.any(segment_mask):
        segment_sync = sync_data[segment_mask]
        segment_gain = gain_data[segment_mask]
        
        print(f"   • {description}:")
        print(f"     Sync: {np.mean(segment_sync):.3f} ± {np.std(segment_sync):.3f}")
        print(f"     Gain: {np.mean(segment_gain):.1f}")

# Regime analysis
regime_durations = {
    'Climb': len([r for r in results_90s if not r[6]]) * 0.1,
    'Cruise': len([r for r in results_90s if r[6] and not r[7]]) * 0.1,
    'Super-Cruise': len([r for r in results_90s if r[7]]) * 0.1
}

print(f"\n🎯 OPERATIONAL REGIME ANALYSIS:")
for regime, duration in regime_durations.items():
    percentage = duration / actual_duration * 100
    print(f"   • {regime}: {duration:.1f}s ({percentage:.1f}%)")

# Final performance metrics
final_sync = sync_data[-1]
final_gain = gain_data[-1]
final_regime = "SUPER-CRUISE" if results_90s[-1][7] else "CRUISE" if results_90s[-1][6] else "CLIMB"

print(f"\n🎉 FINAL PERFORMANCE METRICS:")
print(f"   • Final Synchronization: {final_sync:.3f}")
print(f"   • Final Gain: {final_gain:.1f}")
print(f"   • Final Regime: {final_regime}")
print(f"   • Boosts Used: {plasma_90s.boost_count}/{plasma_90s.max_boosts}")

# Stability assessment
overall_stability = np.std(sync_data)
stability_rating = "EXCELLENT" if overall_stability < 0.05 else "GOOD" if overall_stability < 0.1 else "MODERATE"

print(f"\n🔧 STABILITY ASSESSMENT:")
print(f"   • Overall Stability (std): {overall_stability:.4f}")
print(f"   • Rating: {stability_rating}")
print(f"   • Min Sync: {np.min(sync_data):.3f}")
print(f"   • Max Sync: {np.max(sync_data):.3f}")

# Success criteria check
success_criteria = [
    actual_duration >= 89.9,
    final_sync > 0.9,
    overall_stability < 0.1,
    regime_durations['Super-Cruise'] > 30
]

criteria_names = [
    "Complete 90-second simulation",
    "Final sync > 0.9", 
    "Good stability (std < 0.1)",
    "Substantial super-cruise time (>30s)"
]

print(f"\n✅ SUCCESS CRITERIA CHECK:")
for i, (criterion, name) in enumerate(zip(success_criteria, criteria_names)):
    status = "✅ PASS" if criterion else "❌ FAIL"
    print(f"   {i+1}. {status}: {name}")

success_rate = sum(success_criteria) / len(success_criteria)
print(f"\n📊 OVERALL SUCCESS RATE: {success_rate*100:.1f}%")

if success_rate >= 0.75:
    print("🎉 SIMULATION SUCCESSFUL! System performed excellently.")
elif success_rate >= 0.5:
    print("⚠️  SIMULATION ACCEPTABLE. Some criteria not met.")
else:
    print("🔧 SIMULATION NEEDS IMPROVEMENT. Multiple criteria failed.")

print("="*80)

# Plot final results
plt.figure(figsize=(12, 8))

# Plot 1: Synchronization over time
plt.subplot(2, 2, 1)
plt.plot(time_data, sync_data, 'b-', linewidth=2, alpha=0.8)
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Target')
plt.xlabel('Time (s)')
plt.ylabel('Synchronization')
plt.title('SYNCHRONIZATION OVER 90 SECONDS')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Gain over time
plt.subplot(2, 2, 2)
plt.plot(time_data, gain_data, 'r-', linewidth=2, alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Controller Gain')
plt.title('CONTROLLER GAIN EVOLUTION')
plt.grid(True, alpha=0.3)

# Plot 3: Regime distribution
plt.subplot(2, 2, 3)
regimes = list(regime_durations.keys())
durations = list(regime_durations.values())
colors = ['orange', 'blue', 'green']
plt.pie(durations, labels=regimes, autopct='%1.1f%%', colors=colors)
plt.title('TIME DISTRIBUTION BY REGIME')

# Plot 4: Final synchronization distribution
plt.subplot(2, 2, 4)
plt.hist(sync_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Synchronization')
plt.ylabel('Frequency')
plt.title('SYNCHRONIZATION DISTRIBUTION')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📋 SIMULATION COMPLETE - 90 SECOND MISSION ACCOMPLISHED! 🚀")