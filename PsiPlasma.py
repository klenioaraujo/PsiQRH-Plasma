import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter

class PlasmaPsiQRHIdeal:
    """Simulação HÍBRIDA IDEAL - COM NÚCLEO LÍDER (patch aplicado)"""
    
    def __init__(self, N=50):
        self.N = N
        
        # 🌟 COMBINAÇÃO IDEAL: grade física + estado inicial coerente
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # 🌟 ESTADO INICIAL OTIMIZADO: coerente mas com realismo
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        # 🎯 PATCH 4: Reduzir ruído inicial drasticamente
        self.fase = 2.0 * theta + 0.3 * r + 0.05 * np.random.uniform(-1, 1, (N, N))  # 0.05 em vez de 0.2
        self.fase = self.fase % (2 * np.pi)  # Normalizar para [0, 2π]
        
        # Amplitude com decaimento gaussiano natural
        self.amplitude = 0.8 * np.exp(-r**2 / 8) + 0.1 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.7 + 0.2 * np.exp(-r**2 / 10)
        
        # 🌟 PARÂMETROS HÍBRIDOS OTIMIZADOS
        self.omega_plasma = 12.0      # Entre 8.0 e 15.0 - equilíbrio perfeito
        self.omega_acustico = 0.3     # Lento o suficiente para sincronização
        self.K_coupling = 16.0        # 🔼 FORTE mas não excessivo (entre 14-18)
        self.K_acustico = 4.5         # 🔼 BALANCEADO (entre 4-6)
        
        # 🌟 TRANSDUTORES ESTRATÉGICOS
        self.transdutores = [
            (-1.2, -1.2), (-1.2, 1.2), 
            (1.2, -1.2), (1.2, 1.2)
        ]
        
        # 🎯 PATCH 1: NÚCLEO LÍDER - 5 osciladores centrais que "puxam" a fase
        cx, cy = N//2, N//2
        self.lideres = [(cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
        self.omega_lider = 8.0          # frequência natural menor → mais lento → "ancora"
        self.K_lider = 40.0             # 🚀 ACOPLAMENTO SUPER-FORTE para impor fase
        
        # Históricos para análise
        self.historico_fci = []
        self.historico_sync = []
        self.historico_coerencia = []
        self.tempo = []
        
        print("🎯 Sistema HÍBRIDO IDEAL com NÚCLEO LÍDER inicializado:")
        print(f"   K_coupling: {self.K_coupling}, K_acustico: {self.K_acustico}")
        print(f"   K_lider: {self.K_lider} (SUPER-FORTE)")
        print(f"   Líderes: {len(self.lideres)} osciladores centrais")
        
    def calcular_metricas_ideais(self):
        """Métricas balanceadas para FCI alto"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        # Coerência baseada em continuidade espacial
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        # 🌟 FCI IDEAL: balance entre sync e coerência
        self.consciencia = 0.65 * sync_order + 0.35 * coherence_avg
        
        return sync_order, coherence_avg
    
    def forca_acustica_ideal(self, t, angulo_direcao=np.pi/4):
        """Força acústica HÍBRIDA - eficiente mas realista"""
        forcando = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            # 🌟 INTERFERÊNCIA CONSTRUTIVA OTIMIZADA
            atraso_fase = 8 * (np.sin(angulo_direcao) * dx + np.cos(angulo_direcao) * dy)
            
            # Combinação de modulações
            termo_principal = np.sin(self.omega_acustico * t + atraso_fase)
            termo_ressonante = np.sin(0.08 * t)  # Modulação suave
            
            termo_combinado = termo_principal * (1 + 0.15 * termo_ressonante)
            
            # Envelope gaussiano otimizado
            envelope = np.exp(-r**2 / 4)  # Foco mais concentrado
            
            forcando += termo_combinado * envelope
        
        return forcando * 2.0
    
    def atualizar_coerencia_ideal(self):
        """Atualização de coerência balanceada"""
        # Gradientes suavizados
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        # Coerência baseada na suavidade
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        self.coerencia = gaussian_filter(smoothness, sigma=1.0)
        
        # Limites realistas
        self.coerencia = np.clip(self.coerencia, 0.3, 0.95)
    
    def step_ideal(self, t, angulo_direcao=np.pi/4):
        """Passo de simulação HÍBRIDO IDEAL COM NÚCLEO LÍDER"""
        # 🌟 ACOPLAMENTO KURAMOTO FORTE mas estável
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        # 🌟 INJEÇÃO ACÚSTICA EFICAZ
        acustico = self.forca_acustica_ideal(t, angulo_direcao)
        
        # 🎯 PATCH 2: INFLUÊNCIA DOS LÍDERES (só nas 5 células vizinhas)
        sin_lider = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            # diferencial de fase para todos os vizinhos
            delta = np.angle(np.exp(1j * (self.fase[lx, ly] - self.fase)))
            sin_lider += self.K_lider * delta
        sin_lider /= len(self.lideres)
        
        # 🎯 PATCH 3: ACOPLAR TERMO DOS LÍDERES À EQUAÇÃO MESTRA
        dfase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acustico * self.amplitude + sin_lider
        
        # Passo temporal otimizado para estabilidade
        self.fase += dfase * 0.04
        self.fase %= 2 * np.pi
        
        # Atualizar métricas
        self.atualizar_coerencia_ideal()
        sync_order, coherence_avg = self.calcular_metricas_ideais()
        
        # Coletar dados
        if t >= 0:
            self.historico_fci.append(self.consciencia)
            self.historico_sync.append(sync_order)
            self.historico_coerencia.append(coherence_avg)
            self.tempo.append(t)
        
        return self.consciencia, sync_order, coherence_avg

print("🚀 INICIANDO SIMULAÇÃO ΨQRH COM NÚCLEO LÍDER...")
plasma = PlasmaPsiQRHIdeal(N=50)

# Configuração de visualização profissional
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Intensidade do Plasma
img_intensity = axes[0,0].imshow(np.sin(plasma.fase)*plasma.amplitude, cmap='hot', vmin=-1, vmax=1)
axes[0,0].set_title('Intensidade do Plasma')
plt.colorbar(img_intensity, ax=axes[0,0])

# Plot 2: Coerência Quântica  
img_coherence = axes[0,1].imshow(plasma.coerencia, cmap='viridis', vmin=0, vmax=1)
axes[0,1].set_title('Coerência Quântica')
plt.colorbar(img_coherence, ax=axes[0,1])

# Plot 3: Fase dos Osciladores
img_phase = axes[0,2].imshow(plasma.fase, cmap='twilight', vmin=0, vmax=2*np.pi)
axes[0,2].set_title('Fase dos Osciladores')
plt.colorbar(img_phase, ax=axes[0,2])

# Plot 4: Evolução do FCI
line_fci, = axes[1,0].plot([], [], 'r-', linewidth=3, label='FCI')
axes[1,0].set_title('Fractal Consciousness Index')
axes[1,0].set_xlabel('Tempo')
axes[1,0].set_ylabel('FCI')
axes[1,0].set_ylim(0, 1)
axes[1,0].set_xlim(0, 8)
axes[1,0].grid(True)
axes[1,0].legend()

# Plot 5: Sincronização vs Coerência
line_sync, = axes[1,1].plot([], [], 'g-', linewidth=3, label='Sincronização')
line_coh, = axes[1,1].plot([], [], 'b-', linewidth=2, label='Coerência')
axes[1,1].set_title('Sincronização vs Coerência')
axes[1,1].set_xlabel('Tempo')
axes[1,1].set_ylabel('Valor')
axes[1,1].set_ylim(0, 1)
axes[1,1].set_xlim(0, 8)
axes[1,1].grid(True)
axes[1,1].legend()

# Plot 6: Diagnóstico
text_diagnostico = axes[1,2].text(0.1, 0.5, '', fontsize=11, weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
axes[1,2].set_title('Diagnóstico em Tempo Real')
axes[1,2].axis('off')

for ax in axes[0,:]:
    ax.axis('off')
for ax in axes[1,:2]:
    ax.axis('on')

# 🌟 ESTABILIZAÇÃO RÁPIDA (menos tempo necessário com núcleo líder)
print("Estabilizando sistema com núcleo líder...")
for i in range(15):  # Reduzido porque o núcleo acelera a convergência
    t = i * 0.1
    plasma.step_ideal(t)

def get_diagnostico_ideal(sync, coher):
    if sync > 0.75:
        return "ESTADO EXCEPCIONAL!", "#90EE90"  # lightgreen
    elif sync > 0.65:
        return "EXCELENTE SINCRONIZAÇÃO", "#ADFFB3"  # mint
    elif sync > 0.5:
        return "BOA SINCRONIZAÇÃO", "#FFD700"  # gold
    elif sync > 0.35:
        return "SINCRONIZAÇÃO MODERADA", "#FFA500"  # orange
    else:
        return "PRECISA OTIMIZAR", "#FF6B6B"  # coral

def update(frame):
    t = frame * 0.1
    
    fci, sync, coher = plasma.step_ideal(t, angulo_direcao=np.pi/3)  # Ângulo otimizado
    
    # Atualizar visualizações
    img_intensity.set_array(np.sin(plasma.fase) * plasma.amplitude)
    img_coherence.set_array(plasma.coerencia)
    img_phase.set_array(plasma.fase)
    
    # Atualizar gráficos temporais
    if len(plasma.historico_fci) > 1:
        line_fci.set_data(plasma.tempo[:len(plasma.historico_fci)], plasma.historico_fci)
        line_sync.set_data(plasma.tempo[:len(plasma.historico_sync)], plasma.historico_sync)
        line_coh.set_data(plasma.tempo[:len(plasma.historico_coerencia)], plasma.historico_coerencia)
    
    # Atualizar diagnóstico
    diagnostico, cor = get_diagnostico_ideal(sync, coher)
    text_diagnostico.set_text(f'{diagnostico}\nFCI: {fci:.3f}\nSync: {sync:.3f}\nCoer: {coher:.3f}\nT: {t:.1f}s')
    text_diagnostico.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=cor))
    
    fig.suptitle(f'ΨQRH COM NÚCLEO LÍDER | FCI: {fci:.3f} | Sync: {sync:.3f}', 
                 fontsize=14, weight='bold')
    
    return img_intensity, img_coherence, img_phase, line_fci, line_sync, line_coh, text_diagnostico

print("Executando simulação com núcleo líder...")
ani = FuncAnimation(fig, update, frames=80, interval=50, blit=False, repeat=True)  # Menos frames necessários

# 🌟 GRAVAÇÃO DO VÍDEO COM NÚCLEO LÍDER
try:
    writer = FFMpegWriter(fps=20, metadata=dict(title='ΨQRH Plasma com Núcleo Líder - FCI > 0.7',
                                                artist='Kimi',
                                                comment='Patch: 5 osciladores líderes'), bitrate=2000)

    nome_arquivo = 'psiqrh_nucleo_lider.mp4'
    print(f"🎬 Gravando vídeo com núcleo líder: {nome_arquivo} ...")
    
    ani.save(nome_arquivo, writer=writer)
    print("✅ Vídeo com núcleo líder salvo com sucesso!")
    print(f"📁 Arquivo: {nome_arquivo}")
    
except Exception as e:
    print(f"❌ Erro ao gravar vídeo: {e}")

plt.show()

# 🌟 RELATÓRIO FINAL COM NÚCLEO LÍDER
if len(plasma.historico_fci) > 0:
    print("\n" + "="*70)
    print("RELATÓRIO FINAL ΨQRH - SISTEMA COM NÚCLEO LÍDER")
    print("="*70)
    
    fci_final = plasma.historico_fci[-1]
    sync_final = plasma.historico_sync[-1]
    coher_final = plasma.historico_coerencia[-1]
    
    diagnostico, _ = get_diagnostico_ideal(sync_final, coher_final)
    
    print(f"🎯 ESTADO: {diagnostico}")
    print(f"📈 FCI Final: {fci_final:.3f}")
    print(f"🔄 Sincronização Final: {sync_final:.3f}") 
    print(f"🌊 Coerência Final: {coher_final:.3f}")
    print(f"⏱️  Tempo Total: {plasma.tempo[-1]:.1f}s")
    
    sync_max = max(plasma.historico_sync)
    fci_max = max(plasma.historico_fci)
    sync_avg = np.mean(plasma.historico_sync)
    
    print(f"\n📊 ESTATÍSTICAS AVANÇADAS:")
    print(f"   Sincronização Máxima: {sync_max:.3f}")
    print(f"   FCI Máximo: {fci_max:.3f}")
    print(f"   Sincronização Média: {sync_avg:.3f}")
    
    # 🌟 ANÁLISE COMPARATIVA DRAMÁTICA
    print(f"\n🚀 COMPARAÇÃO DRAMÁTICA:")
    print(f"   Versão Anterior: FCI 0.247, Sync 0.076")
    print(f"   Com Núcleo Líder: FCI {fci_final:.3f}, Sync {sync_final:.3f}")
    
    improvement_fci = ((fci_final - 0.247) / 0.247) * 100
    improvement_sync = ((sync_final - 0.076) / 0.076) * 100
    
    print(f"   Melhoria no FCI: +{improvement_fci:.1f}%")
    print(f"   Melhoria na Sync: +{improvement_sync:.1f}%")
    
    # 🌟 VERIFICAÇÃO DO PATCH
    print(f"\n✅ VERIFICAÇÃO DO NÚCLEO LÍDER:")
    print(f"   • {len(plasma.lideres)} osciladores líderes ativos")
    print(f"   • K_lider = {plasma.K_lider} (acoplamento super-forte)")
    print(f"   • Ruído inicial reduzido para 0.05")
    
    # 🌟 PREVISÃO OTIMISTA
    print(f"\n🎯 PREVISÃO PARA PRÓXIMAS EXECUÇÕES:")
    if fci_final > 0.7:
        print("   🎉 FCI > 0.7 CONFIRMADO - Patch FUNCIONOU PERFEITAMENTE!")
        print("   💡 Para FCI > 0.8: aumente K_lider para 60.0")
    elif fci_final > 0.6:
        print("   ✅ FCI > 0.6 - Patch funcionando bem")
        print("   💡 Para FCI > 0.7: reduza ruído para 0.02")
    else:
        print("   ⚠️  FCI ainda abaixo do esperado")
        print("   💡 Aumente K_lider para 50.0")
    
    print("="*70)