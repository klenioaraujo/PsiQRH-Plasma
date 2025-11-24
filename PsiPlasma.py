import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fft

class PlasmaPsiQRHStable:
    """STABLE VERSION with improved stability and damping"""
    
    def __init__(self, N=50):
        self.N = N
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # More stable initial conditions
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        # 🔥 REDUCE INITIAL NOISE for better stability
        self.fase = 1.5 * theta + 0.2 * r + 0.02 * np.random.uniform(-1, 1, (N, N))
        self.fase = self.fase % (2 * np.pi)
        
        self.amplitude = 0.7 * np.exp(-r**2 / 8) + 0.05 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.7 + 0.2 * np.exp(-r**2 / 10)
        
        # Setpoints
        self.setpoint_sync = 0.70
        self.setpoint_cruzeiro = 0.72
        
        # 🔥 MORE CONSERVATIVE PID BASE VALUES
        self.K_p_base, self.K_i_base, self.K_d_base = 25, 8, 12  # Reduced gains
        
        # System parameters - TUNED FOR STABILITY
        self.omega_plasma = 8.5  # Reduced for better resonance
        self.omega_acustico = 0.3  # Slower acoustic waves
        self.K_coupling = 18.0    # Reduced coupling
        self.K_acustico_base = 3.0  # Reduced acoustic force
        self.K_acustico = self.K_acustico_base
        self.K_lider_base = 45.0   # Reduced leader gain
        self.K_lider = self.K_lider_base
        
        # System states
        self.regime_cruzeiro = False
        self.super_cruzeiro = False
        self.contador_estabilidade = 0
        self.alerta_emitido = False
        self.boost_cooldown = 0  # 🔥 NEW: Prevent constant boosting
        
        self.transdutores = [
            (-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)
        ]
        
        # Leader nucleus
        cx, cy = N//2, N//2
        self.lideres = [
            (cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1),
            (cx-1, cy-1), (cx+1, cy+1), (cx-1, cy+1), (cx+1, cy-1)
        ]
        self.omega_lider = 5.0  # Reduced leader frequency
        
        # Analysis history
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
        
        print("🎯 STABLE ΨQRH SYSTEM:")
        print("   • Conservative PID gains")
        print("   • Boost cooldown system")
        print("   • Reduced initial noise")
        print("   • Improved damping")
        
    def gain_scheduling_pid(self, sync_current):
        """ADAPTIVE PID with MORE CONSERVATIVE settings"""
        if sync_current < 0.4:  # 🔥 LOWER threshold for aggressive mode
            # Aggressive response for very low synchronization
            K_p, K_i, K_d = 35, 12, 15  # 🔥 More derivative for damping
            mode = "AGGRESSIVE"
        elif sync_current < 0.65:  # 🔥 ADJUSTED threshold
            # Balanced for climb phase
            K_p, K_i, K_d = 25, 8, 12
            mode = "BALANCED"
        else:
            # Conservative for high synchronization
            K_p, K_i, K_d = 15, 5, 8  # 🔥 More conservative
            mode = "CONSERVATIVE"
        
        return K_p, K_i, K_d, mode
    
    def detect_oscillations(self):
        """OSCILLATION DETECTION with IMPROVED thresholds"""
        if len(self.historico_sync) >= 15:  # 🔥 Smaller window for faster detection
            # FFT of last 15 sync points
            fft_sync = np.fft.fft(self.historico_sync[-15:])
            magnitudes = np.abs(fft_sync[1:8])  # Focus on lower frequencies
            
            # 🔥 LOWER threshold for oscillation detection
            if np.max(magnitudes) > 3:  # Reduced from 5
                main_freq = np.argmax(magnitudes) + 1
                print(f"⚠️  OSCILLATION DETECTED at freq {main_freq}Hz - Increasing damping")
                return True, np.max(magnitudes)
        
        return False, 0
    
    def alarm_system(self, sync_current, t):
        """IMPROVED ALARM SYSTEM with COOLDOWN"""
        if sync_current < 0.4 and t > 5 and not self.alerta_emitido:
            print("🚨 CRITICAL ALERT: Sync < 0.4! Emergency measures activated!")
            self.K_acustico *= 1.1  # 🔥 SMALLER increase (was 1.2)
            self.K_lider_base *= 1.05  # 🔥 SMALLER increase (was 1.1)
            self.alerta_emitido = True
            return True
        
        # Reset alert if recovered
        if sync_current > 0.6 and self.alerta_emitido:
            print("✅ RECOVERY: Sync normalized")
            self.alerta_emitido = False
            
        return False
    
    def activate_super_cruise(self, sync_current):
        """SUPER-CRUISE MODE with STRICTER conditions"""
        if self.regime_cruzeiro and sync_current > 0.92 and not self.super_cruzeiro:  # 🔥 0.92 instead of 0.95
            print("🌟 SUPER-CRUISE ACTIVATED: sync > 0.92!")
            self.omega_plasma = 7.0  # Even smoother
            self.K_lider *= 0.4      # Minimal control
            self.super_cruzeiro = True
            return True
        return False
    
    def calculate_advanced_metrics(self):
        """Metrics with BETTER smoothing"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        # 🔥 STRONGER smoothing for stability
        if len(self.historico_sync) >= 5:  # Increased from 3
            sync_order = 0.3 * sync_order + 0.7 * np.mean(self.historico_sync[-5:])  # Stronger smoothing
            coherence_avg = 0.3 * coherence_avg + 0.7 * np.mean(self.historico_coerencia[-5:])
        
        self.consciencia = 0.55 * sync_order + 0.45 * coherence_avg
        
        return sync_order, coherence_avg
    
    def advanced_spectral_analysis(self):
        """Spectral analysis with stability focus"""
        try:
            fft_phase = fft2(self.fase)
            fft_magnitude = np.abs(fft_phase)
            
            flattened = fft_magnitude.flatten()
            non_zero_indices = np.where(flattened > 1e-6)[0]
            
            if len(non_zero_indices) > 5:
                dominant_indices = non_zero_indices[np.argsort(flattened[non_zero_indices])[-5:]]
                
                total_energy = np.sum(flattened[non_zero_indices])
                top5_energy = np.sum(flattened[dominant_indices])
                harmony = top5_energy / (total_energy + 1e-8)
                
                # 🔥 MORE CONSERVATIVE coupling adjustment
                adjusted_coupling = 15.0 + 8.0 * harmony  # Reduced range
                self.K_coupling = 0.7 * self.K_coupling + 0.3 * adjusted_coupling  # Slower adjustment
                
                return dominant_indices, harmony
        except:
            pass
        
        return [], 0.5
    
    def advanced_pid_control(self, sync_current, t):
        """PID with STABILITY IMPROVEMENTS"""
        if len(self.historico_sync) < 3:
            return self.K_lider_base, "INIT", 0, 0, 0
        
        # 🔥 COOLDOWN boost system
        if self.boost_cooldown > 0:
            self.boost_cooldown -= 1
        
        # GAIN SCHEDULING
        K_p, K_i, K_d, pid_mode = self.gain_scheduling_pid(sync_current)
        
        current_setpoint = self.setpoint_cruzeiro if self.regime_cruzeiro else self.setpoint_sync
        error = current_setpoint - sync_current
        
        # Integral term with anti-windup
        integral_window = min(6, len(self.historico_sync))  # 🔥 Smaller window
        integral_error = sum(current_setpoint - s for s in self.historico_sync[-integral_window:])
        integral_error = max(-1.5, min(1.5, integral_error))  # 🔥 Tighter limits
        
        # Smoothed derivative term
        if len(self.historico_sync) >= 3:
            derivative_error = (self.historico_sync[-1] - self.historico_sync[-3]) / 2
        else:
            derivative_error = 0
        
        # OSCILLATION DETECTION - adjust K_d if needed
        oscillation_detected, osc_magnitude = self.detect_oscillations()
        if oscillation_detected:
            K_d *= 1.8  # 🔥 STRONGER damping (was 1.5)
            pid_mode = "DAMPED"
        
        # Apply PID
        pid_correction = (K_p * error + K_i * integral_error + K_d * derivative_error)
        
        # Dynamic limits
        upper_limit = 75 if sync_current < 0.5 else 65  # 🔥 Lower limits
        lower_limit = 30 if sync_current > 0.6 else 40
        
        limited_correction = max(-lower_limit, min(upper_limit - self.K_lider_base, pid_correction))
        pid_gain = self.K_lider_base + limited_correction
        
        # 🔥 IMPROVED BOOST with COOLDOWN
        if (sync_current > 0.65 and not self.regime_cruzeiro and 
            self.boost_cooldown == 0 and len(self.historico_sync) > 10):
            # Only boost if recent history shows stability
            recent_sync = self.historico_sync[-10:]
            if np.std(recent_sync) < 0.1:  # Only boost if relatively stable
                pid_gain = min(80, pid_gain + 12)  # 🔥 Smaller boost (was 15)
                pid_mode = "BOOST"
                self.boost_cooldown = 20  # 🔥 2 second cooldown
        
        # ALARM SYSTEM
        self.alarm_system(sync_current, t)
        
        return max(25, min(80, pid_gain)), pid_mode, K_p, K_i, K_d
    
    def detect_advanced_transition(self, sync, coherence):
        """Transition detection with STABILITY CHECKS"""
        if len(self.historico_sync) < 20:  # 🔥 Require more history
            return False
            
        sync_avg = np.mean(self.historico_sync[-15:])  # 🔥 Longer window
        sync_std = np.std(self.historico_sync[-15:])
        
        # 🔥 STRICTER conditions for transition
        main_condition = (sync_avg > 0.70 and  # Increased from 0.68
                         coherence > 0.60 and  # Increased from 0.58
                         sync_std < 0.04 and   # Stricter stability (was 0.06)
                         not self.regime_cruzeiro)
        
        if main_condition:
            self.contador_estabilidade += 1
            if self.contador_estabilidade >= 3:  # 🔥 Require more stable steps (was 2)
                print(f"🚀 TRANSITION DETECTED! Activating cruise mode...")
                self.regime_cruzeiro = True
                self.setpoint_sync = 0.72
                return True
        else:
            self.contador_estabilidade = max(0, self.contador_estabilidade - 1)
            
        return False
    
    def advanced_acoustic_force(self, t, direction_angle=np.pi/4):
        """MORE STABLE acoustic force"""
        forcing = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            phase_delay = 6 * (np.sin(direction_angle) * dx + np.cos(direction_angle) * dy)  # 🔥 Reduced from 8
            main_term = np.sin(self.omega_acustico * t + phase_delay)
            
            if self.super_cruzeiro:
                resonant_term = np.sin(0.008 * t)  # Very smooth in super-cruise
                amplitude = 0.8
            elif self.regime_cruzeiro:
                resonant_term = np.sin(0.012 * t)
                amplitude = 1.0
            else:
                resonant_term = np.sin(0.06 * t)   # 🔥 Slower resonance (was 0.08)
                amplitude = 1.8  # 🔥 Reduced from 2.2
                
            combined_term = main_term * (1 + 0.1 * resonant_term)  # 🔥 Reduced modulation
            envelope = np.exp(-r**2 / 6)  # 🔥 Wider envelope
            
            forcing += combined_term * envelope
        
        return forcing * amplitude * (self.K_acustico / self.K_acustico_base)
    
    def advanced_step(self, t, direction_angle=np.pi/4):
        """Complete step with STABILITY FOCUS"""
        dominant_modes, harmony = self.advanced_spectral_analysis()
        self.historico_harmonia.append(harmony)
        
        # Phase interactions
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        acoustic = self.advanced_acoustic_force(t, direction_angle)
        
        # Leader influence
        sin_leader = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            delta_phase = self.fase[lx, ly] - self.fase
            sin_leader += np.sin(delta_phase)
        
        sync_current_raw = np.abs(np.mean(np.exp(1j * self.fase)))
        
        # ADVANCED PID CONTROL
        pid_gain, pid_mode, K_p, K_i, K_d = self.advanced_pid_control(sync_current_raw, t)
        self.historico_Kp.append(K_p)
        self.historico_Ki.append(K_i)
        self.historico_Kd.append(K_d)
        
        sync_for_detection, coherence_for_detection = self.calculate_advanced_metrics()
        self.detect_advanced_transition(sync_for_detection, coherence_for_detection)
        
        # SUPER-CRUISE
        self.activate_super_cruise(sync_for_detection)
        
        # Regime adjustments
        if self.super_cruzeiro:
            pid_gain *= 0.25  # 🔥 Stronger reduction
            self.K_coupling *= 0.5
        elif self.regime_cruzeiro:
            pid_gain *= 0.5   # 🔥 Stronger reduction
            self.K_coupling *= 0.7
            self.omega_plasma = 7.5
        
        self.K_lider = pid_gain
        sin_leader = self.K_lider * sin_leader / len(self.lideres)
        
        # Master equation
        dphase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acoustic * self.amplitude + sin_leader
        
        # Adaptive DT
        if self.super_cruzeiro:
            dt = 0.012
        elif self.regime_cruzeiro:
            dt = 0.018
        else:
            dt = 0.03 + 0.01 * (1 - sync_current_raw)  # 🔥 Smaller range
        
        self.fase += dphase * dt
        self.fase %= 2 * np.pi
        
        self.update_advanced_coherence()
        sync_order, coherence_avg = self.calculate_advanced_metrics()
        
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

    def update_advanced_coherence(self):
        """Coherence with adaptive smoothing"""
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        if self.super_cruzeiro:
            sigma = 0.6
        elif self.regime_cruzeiro:
            sigma = 1.0
        else:
            sigma = 1.5
            
        self.coerencia = gaussian_filter(smoothness, sigma=sigma)
        self.coerencia = np.clip(self.coerencia, 0.4, 0.9)

# RUN STABLE SIMULATION
print("🚀 STARTING STABLE ΨQRH SIMULATION...")
plasma_stable = PlasmaPsiQRHStable(N=50)

print("Running simulation with stability improvements...")
results = []
special_events = []

for i in range(200):
    t = i * 0.1
    result = plasma_stable.advanced_step(t)
    results.append(result)
    
    fci, sync, coher, gain, dt, harmony, cruise, super_cruise, pid_mode = result
    
    # Record special events
    if pid_mode in ["BOOST", "DAMPED", "AGGRESSIVE"] and i % 15 == 0:
        special_events.append(f"t={t:.1f}s: {pid_mode}")
    
    # Progressive feedback
    if i % 30 == 0 or pid_mode in ["BOOST", "DAMPED"]:
        status = "SUPER-CRUISE 🌟" if super_cruise else "CRUISE ✅" if cruise else f"Sync: {sync:.3f}"
        print(f"t={t:.1f}s | {status} | Gain: {gain:.1f} | PID: {pid_mode}")

# STABILITY REPORT
print("\n" + "="*80)
print("STABILITY REPORT - IMPROVED SYSTEM")
print("="*80)

final_sync = plasma_stable.historico_sync[-1]
final_cruise = plasma_stable.regime_cruzeiro
final_super = plasma_stable.super_cruzeiro

print("🛠️ STABILITY IMPROVEMENTS:")
print("   1. ✅ Conservative PID gains")
print("   2. ✅ Boost cooldown system")
print("   3. ✅ Reduced initial noise")
print("   4. ✅ Improved oscillation damping")
print("   5. ✅ Stricter transition conditions")

print(f"\n📊 FINAL RESULTS:")
print(f"   Synchronization: {final_sync:.3f}")
print(f"   Regime: {'SUPER-CRUISE 🌟' if final_super else 'CRUISE ✅' if final_cruise else 'CLIMB'}")
print(f"   Setpoint: {plasma_stable.setpoint_sync}")

# Stability analysis
sync_history = np.array(plasma_stable.historico_sync)
stability_metric = np.std(sync_history[-50:]) if len(sync_history) >= 50 else np.std(sync_history)

print(f"\n📈 STABILITY ANALYSIS:")
print(f"   Final sync std: {stability_metric:.4f}")
print(f"   Max sync: {np.max(sync_history):.3f}")
print(f"   Min sync: {np.min(sync_history):.3f}")

if final_cruise or final_super:
    print(f"\n🎉 STABILITY ACHIEVED!")
    print(f"   • System reached target regime")
    print(f"   • Stable operation maintained")
else:
    if stability_metric < 0.1:
        print(f"\n📈 GOOD STABILITY: Low oscillations (std: {stability_metric:.4f})")
    else:
        print(f"\n⚠️  NEEDS TUNING: High oscillations (std: {stability_metric:.4f})")

print("="*80)