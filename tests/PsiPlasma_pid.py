import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftfreq

class PlasmaPsiQRHPIDAvancado:
    """Simulação FINAL - COM CONTROLE PID, DETECÇÃO DE TRANSIÇÃO E ANÁLISE ESPECTRAL"""
    
    def __init__(self, N=50):
        self.N = N
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # Estado inicial equilibrado
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        self.fase = 1.5 * theta + 0.2 * r + 0.1 * np.random.uniform(-1, 1, (N, N))
        self.fase = self.fase % (2 * np.pi)
        
        self.amplitude = 0.7 * np.exp(-r**2 / 10) + 0.15 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.6 + 0.3 * np.exp(-r**2 / 12)
        
        # Parâmetros PID
        self.K_p = 25.0  # Proporcional
        self.K_i = 8.0   # Integral  
        self.K_d = 15.0  # Derivativo
        self.setpoint_sync = 0.7  # Alvo de sincronização
        
        # Parâmetros do sistema
        self.omega_plasma = 10.0
        self.omega_acustico = 0.4
        self.K_coupling = 18.0
        self.K_acustico = 3.5
        self.K_lider_base = 55.0
        self.K_lider = self.K_lider_base
        
        # Estados do sistema
        self.regime_cruzeiro = False
        self.transdutores = [
            (-1.2, -1.2), (-1.2, 1.2), 
            (1.2, -1.2), (1.2, 1.2)
        ]
        
        # Núcleo líder
        cx, cy = N//2, N//2
        self.lideres = [(cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
        self.omega_lider = 6.0
        
        # Históricos para controle
        self.historico_fci = []
        self.historico_sync = []
        self.historico_coerencia = []
        self.historico_ganho = []
        self.historico_dt = []
        self.historico_modos = []
        self.tempo = []
        
        print("🎯 SISTEMA ΨQRH COM CONTROLE PID AVANÇADO:")
        print("   • Controle PID completo (P+I+D)")
        print("   • Detecção de transição de fase")
        print("   • Análise espectral em tempo real")
        print("   • Modo cruzeiro automático")
        
    def calcular_metricas_avancadas(self):
        """Métricas com estabilização e análise espectral"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        # Estabilização com média móvel
        if len(self.historico_sync) >= 3:
            sync_order = 0.6 * sync_order + 0.4 * np.mean(self.historico_sync[-3:])
            coherence_avg = 0.6 * coherence_avg + 0.4 * np.mean(self.historico_coerencia[-3:])
        
        self.consciencia = 0.6 * sync_order + 0.4 * coherence_avg
        
        return sync_order, coherence_avg
    
    def analise_espectral_tempo_real(self):
        """Análise FFT para detectar modos dominantes e ajustar acoplamento"""
        # FFT 2D da fase
        fft_phase = fft2(self.fase)
        fft_magnitude = np.abs(fft_phase)
        
        # Encontrar modos dominantes (excluindo DC)
        flattened = fft_magnitude.flatten()
        indices_dominantes = np.argsort(flattened)[-6:-1]  # Top 5 excluindo DC
        
        # Calcular harmonia espectral (inverso da dispersão)
        if len(flattened) > 1:
            harmonia = 1.0 / (np.std(flattened[1:]) + 1e-8)  # Evitar divisão por zero
            harmonia_norm = min(1.0, harmonia / 100)  # Normalizar
        else:
            harmonia_norm = 0.5
        
        # Ajustar K_coupling baseado na harmonia espectral
        coupling_ajustado = 15.0 + 10.0 * harmonia_norm
        self.K_coupling = 0.7 * self.K_coupling + 0.3 * coupling_ajustado
        
        return indices_dominantes, harmonia_norm
    
    def controle_pid_avancado(self, sync_current):
        """Controle PID completo com lógica anti-windup"""
        if len(self.historico_sync) < 2:
            return self.K_lider_base
        
        # Erro atual
        error = self.setpoint_sync - sync_current
        
        # Termo Integral (com janela limitada)
        janela_integral = min(10, len(self.historico_sync))
        integral_error = sum(self.setpoint_sync - s for s in self.historico_sync[-janela_integral:])
        
        # Termo Derivativo
        derivative_error = self.historico_sync[-1] - self.historico_sync[-2]
        
        # Aplicar PID
        correcao_pid = (self.K_p * error + 
                       self.K_i * integral_error + 
                       self.K_d * derivative_error)
        
        # Limitar correção e aplicar anti-windup
        correcao_limitada = max(-30, min(30, correcao_pid))
        ganho_pid = self.K_lider_base + correcao_limitada
        
        return max(20, min(80, ganho_pid))
    
    def detectar_transicao_fase(self, sync, coherence):
        """Detectar quando o sistema entra em regime coerente"""
        if (sync > 0.7 and coherence > 0.6 and 
            not self.regime_cruzeiro and 
            len(self.historico_sync) > 20):
            
            # Verificar estabilidade recente
            sync_recente = self.historico_sync[-10:]
            if np.std(sync_recente) < 0.05:  # Estável
                print("🚀 TRANSIÇÃO DETECTADA: Ativando modo cruzeiro!")
                self.regime_cruzeiro = True
                return True
        return False
    
    def forca_acustica_avancada(self, t, angulo_direcao=np.pi/4):
        """Força acústica com modulação adaptativa"""
        forcando = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            atraso_fase = 6 * (np.sin(angulo_direcao) * dx + np.cos(angulo_direcao) * dy)
            termo_principal = np.sin(self.omega_acustico * t + atraso_fase)
            
            # Modulação baseada no regime
            if self.regime_cruzeiro:
                termo_ressonante = np.sin(0.02 * t)  # Mais suave no cruzeiro
            else:
                termo_ressonante = np.sin(0.05 * t)  # Mais ativo na subida
                
            termo_combinado = termo_principal * (1 + 0.1 * termo_ressonante)
            envelope = np.exp(-r**2 / 6)
            
            forcando += termo_combinado * envelope
        
        return forcando * (1.5 if self.regime_cruzeiro else 1.8)
    
    def atualizar_coerencia_avancada(self):
        """Coerência com suavização adaptativa"""
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        # Suavização baseada no regime
        sigma = 1.0 if self.regime_cruzeiro else 1.5
        self.coerencia = gaussian_filter(smoothness, sigma=sigma)
        self.coerencia = np.clip(self.coerencia, 0.4, 0.9)
    
    def step_avancado(self, t, angulo_direcao=np.pi/4):
        """Passo com todas as técnicas avançadas integradas"""
        # Análise espectral primeiro
        modos_dominantes, harmonia = self.analise_espectral_tempo_real()
        self.historico_modos.append(modos_dominantes)
        
        # Cálculo das interações
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        acustico = self.forca_acustica_avancada(t, angulo_direcao)
        
        # Núcleo líder com controle PID
        sin_lider = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            delta_phase = self.fase[lx, ly] - self.fase
            sin_lider += np.sin(delta_phase)
        
        sync_current_raw = np.abs(np.mean(np.exp(1j * self.fase)))
        
        # Aplicar controle PID
        ganho_pid = self.controle_pid_avancado(sync_current_raw)
        
        # Detecção de transição de fase
        sync_for_detection, coherence_for_detection = self.calcular_metricas_avancadas()
        if self.detectar_transicao_fase(sync_for_detection, coherence_for_detection):
            # Modo cruzeiro - parâmetros conservadores
            ganho_pid *= 0.6
            self.K_coupling *= 0.8
        
        self.K_lider = ganho_pid
        sin_lider = self.K_lider * sin_lider / len(self.lideres)
        
        # Equação mestre com DT adaptativo
        dfase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acustico * self.amplitude + sin_lider
        
        # DT adaptativo baseado no regime
        if self.regime_cruzeiro:
            dt = 0.025  # Mais fino no cruzeiro
        else:
            dt = 0.04 + 0.02 * (1 - sync_current_raw)
        
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
                self.K_lider, dt, harmonia, self.regime_cruzeiro)

# Configuração de visualização expandida
print("🚀 INICIANDO SIMULAÇÃO ΨQRH COM CONTROLE PID AVANÇADO...")
plasma = PlasmaPsiQRHPIDAvancado(N=50)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Configuração dos plots (similar ao anterior, mas com métricas adicionais)
img_intensity = axes[0,0].imshow(np.sin(plasma.fase)*plasma.amplitude, cmap='hot', vmin=-1, vmax=1)
axes[0,0].set_title('Intensidade do Plasma')
plt.colorbar(img_intensity, ax=axes[0,0])

img_coherence = axes[0,1].imshow(plasma.coerencia, cmap='viridis', vmin=0, vmax=1)
axes[0,1].set_title('Coerência Quântica')
plt.colorbar(img_coherence, ax=axes[0,1])

img_phase = axes[0,2].imshow(plasma.fase, cmap='twilight', vmin=0, vmax=2*np.pi)
axes[0,2].set_title('Fase dos Osciladores')
plt.colorbar(img_phase, ax=axes[0,2])

img_harmonia = axes[0,3].imshow(np.ones((50,50)), cmap='plasma', vmin=0, vmax=1)
axes[0,3].set_title('Harmonia Espectral')
plt.colorbar(img_harmonia, ax=axes[0,3])

# Gráficos temporais
line_fci, = axes[1,0].plot([], [], 'r-', linewidth=3, label='FCI')
axes[1,0].set_title('Fractal Consciousness Index')
axes[1,0].set_ylim(0, 1)
axes[1,0].grid(True)
axes[1,0].legend()

line_sync, = axes[1,1].plot([], [], 'g-', linewidth=3, label='Sincronização')
line_coh, = axes[1,1].plot([], [], 'b-', linewidth=2, label='Coerência')
axes[1,1].set_title('Sincronização vs Coerência')
axes[1,1].set_ylim(0, 1)
axes[1,1].grid(True)
axes[1,1].legend()

line_ganho, = axes[1,2].plot([], [], 'm-', linewidth=2, label='Ganho PID')
line_harmonia, = axes[1,2].plot([], [], 'y-', linewidth=2, label='Harmonia')
axes[1,2].set_title('Controle PID e Harmonia')
axes[1,2].set_ylim(0, 100)
axes[1,2].grid(True)
axes[1,2].legend()

text_diagnostico = axes[1,3].text(0.1, 0.5, '', fontsize=10, weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
axes[1,3].set_title('Diagnóstico PID Avançado')
axes[1,3].axis('off')

for ax in axes[0,:]:
    ax.axis('off')
for ax in axes[1,:3]:
    ax.axis('on')

# Inicialização
print("Inicializando sistema com controle PID...")
for i in range(25):
    t = i * 0.1
    fci, sync, coher, ganho, dt, harmonia, regime = plasma.step_avancado(t)
    plasma.historico_ganho.append(ganho)

def get_diagnostico_pid(sync, coher, ganho, harmonia, regime):
    if regime:
        return "MODO CRUZEIRO ATIVO!", "#90EE90"
    elif sync > 0.7:
        return "ALTA SINCRONIZAÇÃO", "#ADFFB3" 
    elif harmonia > 0.7:
        return "ALTA HARMONIA", "#FFD700"
    elif sync > 0.55:
        return "SUBINDO PARA CRUZEIRO", "#FFA500"
    else:
        return "FASE DE ACELERAÇÃO", "#FF6B6B"

def update(frame):
    t = frame * 0.1
    
    fci, sync, coher, ganho, dt, harmonia, regime = plasma.step_avancado(t, angulo_direcao=np.pi/4)
    
    # Atualizar visualizações
    img_intensity.set_array(np.sin(plasma.fase) * plasma.amplitude)
    img_coherence.set_array(plasma.coerencia)
    img_phase.set_array(plasma.fase)
    img_harmonia.set_array(np.ones((50,50)) * harmonia)
    
    if len(plasma.historico_fci) > 1:
        line_fci.set_data(plasma.tempo[:len(plasma.historico_fci)], plasma.historico_fci)
        line_sync.set_data(plasma.tempo[:len(plasma.historico_sync)], plasma.historico_sync)
        line_coh.set_data(plasma.tempo[:len(plasma.historico_coerencia)], plasma.historico_coerencia)
        line_ganho.set_data(plasma.tempo[:len(plasma.historico_ganho)], plasma.historico_ganho)
        line_harmonia.set_data(plasma.tempo[:len(plasma.historico_modos)], 
                              [h*80 for h in plasma.historico_modos])  # Escala
    
    diagnostico, cor = get_diagnostico_pid(sync, coher, ganho, harmonia, regime)
    text_diagnostico.set_text(
        f'{diagnostico}\n'
        f'FCI: {fci:.3f}\nSync: {sync:.3f}\nCoer: {coher:.3f}\n'
        f'Ganho: {ganho:.1f}\nHarmonia: {harmonia:.3f}\n'
        f'Regime: {"CRUZEIRO" if regime else "SUBIDA"}\nT: {t:.1f}s'
    )
    text_diagnostico.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=cor))
    
    fig.suptitle(f'ΨQRH CONTROLE PID AVANÇADO | FCI: {fci:.3f} | Sync: {sync:.3f} | Harmonia: {harmonia:.3f}', 
                 fontsize=14, weight='bold')
    
    return (img_intensity, img_coherence, img_phase, img_harmonia, 
            line_fci, line_sync, line_coh, line_ganho, line_harmonia, text_diagnostico)

# Executar simulação
print("Executando simulação com controle PID avançado...")
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False, repeat=True)

plt.show()

# RELATÓRIO FINAL
if len(plasma.historico_fci) > 0:
    print("\n" + "="*75)
    print("RELATÓRIO FINAL ΨQRH - CONTROLE PID AVANÇADO")
    print("="*75)
    
    fci_final = plasma.historico_fci[-1]
    sync_final = plasma.historico_sync[-1]
    ganho_final = plasma.historico_ganho[-1]
    
    print(f"🎯 ESTADO FINAL:")
    print(f"   FCI: {fci_final:.3f}, Sincronização: {sync_final:.3f}")
    print(f"   Ganho PID: {ganho_final:.1f}, Regime: {'CRUZEIRO' if plasma.regime_cruzeiro else 'SUBIDA'}")
    
    print(f"\n⚙️  CONTROLE PID:")
    print(f"   Kp: {plasma.K_p}, Ki: {plasma.K_i}, Kd: {plasma.K_d}")
    print(f"   Setpoint: {plasma.setpoint_sync}")
    
    if plasma.regime_cruzeiro:
        print(f"\n🚀 TRANSIÇÃO BEM-SUCEDIDA: Sistema em modo cruzeiro!")
        print(f"   • Ganho reduzido automaticamente")
        print(f"   • DT mais fino (0.025)")
        print(f"   • Acoplamento otimizado")
    else:
        print(f"\n📈 SISTEMA EM FASE DE SUBIDA:")
        print(f"   • Controle PID ativo")
        print(f"   • Buscando transição para cruzeiro")
    
    print("="*75)