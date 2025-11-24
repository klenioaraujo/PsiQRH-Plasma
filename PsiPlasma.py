import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fft

class PlasmaPsiQRHPIDAvancado:
    """ADVANCED VERSION with all proposed improvements"""
    
    def __init__(self, N=50):
        self.N = N
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # Estado inicial
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        self.fase = 1.5 * theta + 0.2 * r + 0.05 * np.random.uniform(-1, 1, (N, N))
        self.fase = self.fase % (2 * np.pi)
        
        self.amplitude = 0.7 * np.exp(-r**2 / 8) + 0.1 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.7 + 0.2 * np.exp(-r**2 / 10)
        
        # Setpoints
        self.setpoint_sync = 0.70
        self.setpoint_cruzeiro = 0.72
        
        # ADAPTIVE PID (Gain Scheduling)
        self.K_p_base, self.K_i_base, self.K_d_base = 30, 12, 10  # Valores base
        
        # Parâmetros do sistema
        self.omega_plasma = 9.0
        self.omega_acustico = 0.35
        self.K_coupling = 22.0
        self.K_acustico_base = 4.0
        self.K_acustico = self.K_acustico_base
        self.K_lider_base = 60.0
        self.K_lider = self.K_lider_base
        
        # Estados do sistema
        self.regime_cruzeiro = False
        self.super_cruzeiro = False
        self.contador_estabilidade = 0
        self.alerta_emitido = False
        
        self.transdutores = [
            (-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)
        ]
        
        # Núcleo líder
        cx, cy = N//2, N//2
        self.lideres = [
            (cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1),
            (cx-1, cy-1), (cx+1, cy+1), (cx-1, cy+1), (cx+1, cy-1)
        ]
        self.omega_lider = 5.5
        
        # Históricos para análise
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
        
        print("🎯 ΨQRH SYSTEM WITH ADVANCED CONTROL:")
        print("   • Adaptive PID (Gain Scheduling)")
        print("   • Oscillation Detection by FFT")
        print("   • Critical Alarm System")
        print("   • Super-Cruise Mode")
        
    def gain_scheduling_pid(self, sync_current):
        """ADAPTIVE PID - adjusts gains based on sync level"""
        if sync_current < 0.5:
            # Aggressive response for low synchronization
            K_p, K_i, K_d = 40, 15, 12
            modo = "AGGRESSIVE"
        elif sync_current < 0.7:
            # Balanced for climb phase
            K_p, K_i, K_d = 30, 12, 10
            modo = "BALANCED"
        else:
            # Conservative for high synchronization
            K_p, K_i, K_d = 20, 8, 6
            modo = "CONSERVATIVE"
        
        return K_p, K_i, K_d, modo
    
    def detectar_oscilacoes(self):
        """OSCILLATION DETECTION by spectral analysis"""
        if len(self.historico_sync) >= 20:
            # FFT dos últimos 20 pontos de sync
            fft_sync = np.fft.fft(self.historico_sync[-20:])
            magnitudes = np.abs(fft_sync[1:10])  # Ignorar DC e altas frequências
            
            # 🔥 Detectar oscilações persistentes
            if np.max(magnitudes) > 5:  # Threshold para oscilação forte
                freq_principal = np.argmax(magnitudes) + 1
                print(f"⚠️  DETECTED: Oscillation at freq {freq_principal}Hz - Increasing damping")
                return True, np.max(magnitudes)
        
        return False, 0
    
    def sistema_alarme(self, sync_current, t):
        """ALARM SYSTEM for critical conditions"""
        if sync_current < 0.4 and t > 5 and not self.alerta_emitido:
            print("🚨 CRITICAL ALERT: Synchronization below 0.4! Activating emergency measures!")
            self.K_acustico *= 1.2  # Aumenta força acústica
            self.K_lider_base *= 1.1  # Aumenta ganho base
            self.alerta_emitido = True
            return True
        
        # Resetar alerta se recuperou
        if sync_current > 0.6 and self.alerta_emitido:
            print("✅ RECOVERY: Synchronization normalized")
            self.alerta_emitido = False
            
        return False
    
    def ativar_super_cruzeiro(self, sync_current):
        """SUPER-CRUISE MODE for extreme performance"""
        if self.regime_cruzeiro and sync_current > 0.95 and not self.super_cruzeiro:
            print("🌟 ACTIVATING SUPER-CRUISE: sync > 0.95!")
            self.omega_plasma = 7.5  # Ainda mais suave
            self.K_lider *= 0.5      # Controle mínimo
            self.super_cruzeiro = True
            return True
        return False
    
    def calcular_metricas_avancadas(self):
        """Metrics with fast smoothing"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        if len(self.historico_sync) >= 3:
            sync_order = 0.4 * sync_order + 0.6 * np.mean(self.historico_sync[-3:])
            coherence_avg = 0.4 * coherence_avg + 0.6 * np.mean(self.historico_coerencia[-3:])
        
        self.consciencia = 0.55 * sync_order + 0.45 * coherence_avg
        
        return sync_order, coherence_avg
    
    def analise_espectral_avancada(self):
        """Spectral analysis with harmony detection"""
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
                
                coupling_ajustado = 18.0 + 12.0 * harmonia
                self.K_coupling = 0.6 * self.K_coupling + 0.4 * coupling_ajustado
                
                return indices_dominantes, harmonia
        except:
            pass
        
        return [], 0.5
    
    def controle_pid_avancado(self, sync_current, t):
        """PID with all advanced features"""
        if len(self.historico_sync) < 3:
            return self.K_lider_base, "INIT", 0, 0, 0
        
        # GAIN SCHEDULING - selects PID gains
        K_p, K_i, K_d, modo_pid = self.gain_scheduling_pid(sync_current)
        
        setpoint_atual = self.setpoint_cruzeiro if self.regime_cruzeiro else self.setpoint_sync
        error = setpoint_atual - sync_current
        
        # Termo Integral com anti-windup
        janela_integral = min(8, len(self.historico_sync))
        integral_error = sum(setpoint_atual - s for s in self.historico_sync[-janela_integral:])
        integral_error = max(-2, min(2, integral_error))
        
        # Termo Derivativo suavizado
        if len(self.historico_sync) >= 3:
            derivative_error = (self.historico_sync[-1] - self.historico_sync[-3]) / 2
        else:
            derivative_error = 0
        
        # OSCILLATION DETECTION - adjusts K_d if necessary
        oscilacao_detectada, magnitude_osc = self.detectar_oscilacoes()
        if oscilacao_detectada:
            K_d *= 1.5  # Increases damping
            modo_pid = "DAMPED"
        
        # Aplicar PID
        correcao_pid = (K_p * error + K_i * integral_error + K_d * derivative_error)
        
        # Limites dinâmicos
        limite_superior = 80 if sync_current < 0.6 else 70
        limite_inferior = 25 if sync_current > 0.65 else 35
        
        correcao_limitada = max(-limite_inferior, min(limite_superior - self.K_lider_base, correcao_pid))
        ganho_pid = self.K_lider_base + correcao_limitada
        
        # TEMPORARY BOOST
        if sync_current > 0.65 and not self.regime_cruzeiro:
            ganho_pid = min(85, ganho_pid + 15)
            modo_pid = "BOOST"
        
        # ALARM SYSTEM
        self.sistema_alarme(sync_current, t)
        
        return max(20, min(85, ganho_pid)), modo_pid, K_p, K_i, K_d
    
    def detectar_transicao_avancada(self, sync, coherence):
        """Transition detection with reduced counter"""
        if len(self.historico_sync) < 15:
            return False
            
        sync_media = np.mean(self.historico_sync[-10:])
        sync_std = np.std(self.historico_sync[-10:])
        
        condicao_principal = (sync_media > 0.68 and
                            coherence > 0.58 and  
                            sync_std < 0.06 and
                            not self.regime_cruzeiro)
        
        if condicao_principal:
            self.contador_estabilidade += 1
            if self.contador_estabilidade >= 2:
                print(f"🚀 TRANSITION DETECTED! Activating cruise mode...")
                self.regime_cruzeiro = True
                self.setpoint_sync = 0.72
                return True
        else:
            self.contador_estabilidade = max(0, self.contador_estabilidade - 1)
            
        return False
    
    def forca_acustica_avancada(self, t, angulo_direcao=np.pi/4):
        """Adaptive acoustic force"""
        forcando = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            atraso_fase = 8 * (np.sin(angulo_direcao) * dx + np.cos(angulo_direcao) * dy)
            termo_principal = np.sin(self.omega_acustico * t + atraso_fase)
            
            if self.super_cruzeiro:
                termo_ressonante = np.sin(0.01 * t)  # Muito suave no super-cruzeiro
                amplitude = 1.0
            elif self.regime_cruzeiro:
                termo_ressonante = np.sin(0.015 * t)
                amplitude = 1.2
            else:
                termo_ressonante = np.sin(0.08 * t)
                amplitude = 2.2
                
            termo_combinado = termo_principal * (1 + 0.15 * termo_ressonante)
            envelope = np.exp(-r**2 / 5)
            
            forcando += termo_combinado * envelope
        
        return forcando * amplitude * (self.K_acustico / self.K_acustico_base)
    
    def step_avancado(self, t, angulo_direcao=np.pi/4):
        """Complete step with all features"""
        modos_dominantes, harmonia = self.analise_espectral_avancada()
        self.historico_harmonia.append(harmonia)
        
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        acustico = self.forca_acustica_avancada(t, angulo_direcao)
        
        sin_lider = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            delta_phase = self.fase[lx, ly] - self.fase
            sin_lider += np.sin(delta_phase)
        
        sync_current_raw = np.abs(np.mean(np.exp(1j * self.fase)))
        
        # ADVANCED PID CONTROL with all features
        ganho_pid, modo_pid, K_p, K_i, K_d = self.controle_pid_avancado(sync_current_raw, t)
        self.historico_Kp.append(K_p)
        self.historico_Ki.append(K_i)
        self.historico_Kd.append(K_d)
        
        sync_for_detection, coherence_for_detection = self.calcular_metricas_avancadas()
        self.detectar_transicao_avancada(sync_for_detection, coherence_for_detection)
        
        # SUPER-CRUISE
        self.ativar_super_cruzeiro(sync_for_detection)
        
        # Ajustes de regime
        if self.super_cruzeiro:
            ganho_pid *= 0.3
            self.K_coupling *= 0.6
        elif self.regime_cruzeiro:
            ganho_pid *= 0.55
            self.K_coupling *= 0.75
            self.omega_plasma = 8.0
        
        self.K_lider = ganho_pid
        sin_lider = self.K_lider * sin_lider / len(self.lideres)
        
        dfase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acustico * self.amplitude + sin_lider
        
        # DT adaptativo
        if self.super_cruzeiro:
            dt = 0.015
        elif self.regime_cruzeiro:
            dt = 0.02
        else:
            dt = 0.035 + 0.015 * (1 - sync_current_raw)
        
        self.fase += dfase * dt
        self.fase %= 2 * np.pi
        
        self.atualizar_coerencia_avancada()
        sync_order, coherence_avg = self.calcular_metricas_avancadas()
        
        if t >= 0:
            self.historico_fci.append(self.consciencia)
            self.historico_sync.append(sync_order)
            self.historico_coerencia.append(coherence_avg)
            self.historico_ganho.append(self.K_lider)
            self.historico_dt.append(dt)
            self.tempo.append(t)
        
        return (self.consciencia, sync_order, coherence_avg, 
                self.K_lider, dt, harmonia, self.regime_cruzeiro, 
                self.super_cruzeiro, modo_pid)

    def atualizar_coerencia_avancada(self):
        """Coherence with adaptive smoothing"""
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        if self.super_cruzeiro:
            sigma = 0.8
        elif self.regime_cruzeiro:
            sigma = 1.2
        else:
            sigma = 1.8
            
        self.coerencia = gaussian_filter(smoothness, sigma=sigma)
        self.coerencia = np.clip(self.coerencia, 0.4, 0.9)

# EXECUTAR SIMULAÇÃO AVANÇADA
print("🚀 STARTING ΨQRH SIMULATION WITH ADVANCED CONTROL...")
plasma_avancado = PlasmaPsiQRHPIDAvancado(N=50)

print("Running simulation with all advanced features...")
resultados = []
eventos_especiais = []

for i in range(200):
    t = i * 0.1
    resultado = plasma_avancado.step_avancado(t)
    resultados.append(resultado)
    
    fci, sync, coher, ganho, dt, harmonia, regime, super_cruzeiro, modo_pid = resultado
    
    # Registrar eventos especiais
    if modo_pid in ["BOOST", "DAMPED", "AGGRESSIVE"] and i % 10 == 0:
        eventos_especiais.append(f"t={t:.1f}s: {modo_pid}")
    
    # Feedback progressivo
    if i % 25 == 0 or modo_pid in ["BOOST", "DAMPED"]:
        status = "SUPER-CRUISE 🌟" if super_cruzeiro else "CRUISE ✅" if regime else f"Sync: {sync:.3f}"
        print(f"t={t:.1f}s | {status} | Gain: {ganho:.1f} | PID: {modo_pid}")

# ADVANCED REPORT
print("\n" + "="*80)
print("FINAL REPORT - ADVANCED CONTROL")
print("="*80)

sync_final = plasma_avancado.historico_sync[-1]
regime_final = plasma_avancado.regime_cruzeiro
super_final = plasma_avancado.super_cruzeiro

print("🎯 IMPLEMENTED FEATURES:")
print("   1. ✅ Adaptive PID (Gain Scheduling)")
print("   2. ✅ Oscillation Detection by FFT")
print("   3. ✅ Critical Alarm System")
print("   4. ✅ Super-Cruise Mode")

print(f"\n📊 FINAL RESULT:")
print(f"   Synchronization: {sync_final:.3f}")
print(f"   Regime: {'SUPER-CRUISE 🌟' if super_final else 'CRUISE ✅' if regime_final else 'CLIMB'}")
print(f"   Setpoint: {plasma_avancado.setpoint_sync}")

print(f"\n🔧 DETECTED SPECIAL EVENTS:")
for evento in eventos_especiais[-5:]:  # Last 5 events
    print(f"   • {evento}")

if super_final:
    print(f"\n🎉 EXCEPTIONAL PERFORMANCE!")
    print(f"   • System reached SUPER-CRUISE mode")
    print(f"   • Extreme synchronization: {sync_final:.3f}")
    print(f"   • Ultra-optimized parameters")
elif regime_final:
    print(f"\n✅ SUCCESS! System stable in cruise mode")
else:
    print(f"\n📈 IN PROGRESS: Sync = {sync_final:.3f}")

print("="*80)