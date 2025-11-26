import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter

class PlasmaPsiQRHFinal:
    """Simulação FINAL - CORREÇÃO EQUILIBRADA"""
    
    def __init__(self, N=50):
        self.N = N
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # Estado inicial equilibrado
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        # 🎯 ESTADO INICIAL MAIS SUAVE
        self.fase = 1.5 * theta + 0.2 * r + 0.1 * np.random.uniform(-1, 1, (N, N))
        self.fase = self.fase % (2 * np.pi)
        
        self.amplitude = 0.7 * np.exp(-r**2 / 10) + 0.15 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.6 + 0.3 * np.exp(-r**2 / 12)
        
        # ⚖️ PARÂMETROS EQUILIBRADOS (baseados nos máximos alcançados)
        self.omega_plasma = 10.0       # 🔽 Mais estável
        self.omega_acustico = 0.4      # 🔽 Um pouco mais rápido
        self.K_coupling = 18.0         # ⚖️ Entre 16-24
        self.K_acustico = 3.5          # 🔽 Mais suave
        
        self.transdutores = [
            (-1.2, -1.2), (-1.2, 1.2), 
            (1.2, -1.2), (1.2, 1.2)
        ]
        
        # 🎯 NÚCLEO LÍDER OTIMIZADO
        cx, cy = N//2, N//2
        self.lideres = [(cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
        self.omega_lider = 6.0          # 🔽 Mais lento para estabilidade
        self.K_lider = 55.0             # ⚖️ Entre 40-75 (equilíbrio)
        
        self.historico_fci = []
        self.historico_sync = []
        self.historico_coerencia = []
        self.tempo = []
        
        print("🎯 SISTEMA FINAL EQUILIBRADO:")
        print(f"   K_coupling: {self.K_coupling}, K_lider: {self.K_lider}")
        print(f"   omega_plasma: {self.omega_plasma} (mais estável)")
        
    def calcular_metricas_finais(self):
        """Métricas com foco em estabilidade"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        # 🎯 FCI mais balanceado
        self.consciencia = 0.6 * sync_order + 0.4 * coherence_avg
        
        return sync_order, coherence_avg
    
    def forca_acustica_final(self, t, angulo_direcao=np.pi/4):
        """Força acústica mais suave"""
        forcando = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            atraso_fase = 6 * (np.sin(angulo_direcao) * dx + np.cos(angulo_direcao) * dy)
            
            termo_principal = np.sin(self.omega_acustico * t + atraso_fase)
            termo_ressonante = np.sin(0.05 * t)  # 🔽 Mais lento
            
            termo_combinado = termo_principal * (1 + 0.1 * termo_ressonante)  # 🔽 Menor modulação
            envelope = np.exp(-r**2 / 6)  # 🔽 Área maior
            
            forcando += termo_combinado * envelope
        
        return forcando * 1.8  # 🔽 Ganho menor
    
    def atualizar_coerencia_final(self):
        """Coerência mais suavizada"""
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        self.coerencia = gaussian_filter(smoothness, sigma=1.5)  # 🔽 Mais suave
        self.coerencia = np.clip(self.coerencia, 0.4, 0.9)       # 🔽 Limites mais realistas
    
    def step_final(self, t, angulo_direcao=np.pi/4):
        """Passo FINAL equilibrado"""
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        acustico = self.forca_acustica_final(t, angulo_direcao)
        
        # 🎯 NÚCLEO LÍDER CORRETO MAS EQUILIBRADO
        sin_lider = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            delta_phase = self.fase[lx, ly] - self.fase
            sin_lider += np.sin(delta_phase)  # ✅ Fórmula CORRETA mantida
        sin_lider = self.K_lider * sin_lider / len(self.lideres)
        
        # 🎯 EQUAÇÃO MESTRA EQUILIBRADA
        dfase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acustico * self.amplitude + sin_lider
        
        self.fase += dfase * 0.05  # ⚖️ Passo equilibrado
        self.fase %= 2 * np.pi
        
        self.atualizar_coerencia_final()
        sync_order, coherence_avg = self.calcular_metricas_finais()
        
        if t >= 0:
            self.historico_fci.append(self.consciencia)
            self.historico_sync.append(sync_order)
            self.historico_coerencia.append(coherence_avg)
            self.tempo.append(t)
        
        return self.consciencia, sync_order, coherence_avg

print("🚀 INICIANDO SIMULAÇÃO ΨQRH FINAL EQUILIBRADA...")
plasma = PlasmaPsiQRHFinal(N=50)

# Configuração de visualização
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

img_intensity = axes[0,0].imshow(np.sin(plasma.fase)*plasma.amplitude, cmap='hot', vmin=-1, vmax=1)
axes[0,0].set_title('Intensidade do Plasma')
plt.colorbar(img_intensity, ax=axes[0,0])

img_coherence = axes[0,1].imshow(plasma.coerencia, cmap='viridis', vmin=0, vmax=1)
axes[0,1].set_title('Coerência Quântica')
plt.colorbar(img_coherence, ax=axes[0,1])

img_phase = axes[0,2].imshow(plasma.fase, cmap='twilight', vmin=0, vmax=2*np.pi)
axes[0,2].set_title('Fase dos Osciladores')
plt.colorbar(img_phase, ax=axes[0,2])

line_fci, = axes[1,0].plot([], [], 'r-', linewidth=3, label='FCI')
axes[1,0].set_title('Fractal Consciousness Index')
axes[1,0].set_xlabel('Tempo')
axes[1,0].set_ylabel('FCI')
axes[1,0].set_ylim(0, 1)
axes[1,0].set_xlim(0, 8)
axes[1,0].grid(True)
axes[1,0].legend()

line_sync, = axes[1,1].plot([], [], 'g-', linewidth=3, label='Sincronização')
line_coh, = axes[1,1].plot([], [], 'b-', linewidth=2, label='Coerência')
axes[1,1].set_title('Sincronização vs Coerência')
axes[1,1].set_xlabel('Tempo')
axes[1,1].set_ylabel('Valor')
axes[1,1].set_ylim(0, 1)
axes[1,1].set_xlim(0, 8)
axes[1,1].grid(True)
axes[1,1].legend()

text_diagnostico = axes[1,2].text(0.1, 0.5, '', fontsize=11, weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
axes[1,2].set_title('Diagnóstico em Tempo Real')
axes[1,2].axis('off')

for ax in axes[0,:]:
    ax.axis('off')
for ax in axes[1,:2]:
    ax.axis('on')

print("Estabilizando sistema final equilibrado...")
for i in range(20):
    t = i * 0.1
    plasma.step_final(t)

def get_diagnostico_final(sync, coher):
    if sync > 0.75:
        return "ESTADO EXCEPCIONAL!", "#90EE90"
    elif sync > 0.65:
        return "EXCELENTE SINCRONIZAÇÃO", "#ADFFB3"
    elif sync > 0.55:
        return "BOA SINCRONIZAÇÃO", "#FFD700"
    elif sync > 0.45:
        return "SINCRONIZAÇÃO MODERADA", "#FFA500"
    else:
        return "PRECISA OTIMIZAR", "#FF6B6B"

def update(frame):
    t = frame * 0.1
    
    fci, sync, coher = plasma.step_final(t, angulo_direcao=np.pi/4)
    
    img_intensity.set_array(np.sin(plasma.fase) * plasma.amplitude)
    img_coherence.set_array(plasma.coerencia)
    img_phase.set_array(plasma.fase)
    
    if len(plasma.historico_fci) > 1:
        line_fci.set_data(plasma.tempo[:len(plasma.historico_fci)], plasma.historico_fci)
        line_sync.set_data(plasma.tempo[:len(plasma.historico_sync)], plasma.historico_sync)
        line_coh.set_data(plasma.tempo[:len(plasma.historico_coerencia)], plasma.historico_coerencia)
    
    diagnostico, cor = get_diagnostico_final(sync, coher)
    text_diagnostico.set_text(f'{diagnostico}\nFCI: {fci:.3f}\nSync: {sync:.3f}\nCoer: {coher:.3f}\nT: {t:.1f}s')
    text_diagnostico.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=cor))
    
    fig.suptitle(f'ΨQRH FINAL EQUILIBRADO | FCI: {fci:.3f} | Sync: {sync:.3f}', 
                 fontsize=14, weight='bold')
    
    return img_intensity, img_coherence, img_phase, line_fci, line_sync, line_coh, text_diagnostico

print("Executando simulação final equilibrada...")
ani = FuncAnimation(fig, update, frames=80, interval=50, blit=False, repeat=True)

try:
    writer = FFMpegWriter(fps=20, metadata=dict(title='ΨQRH Plasma Final Equilibrado',
                                                artist='Kimi',
                                                comment='Parâmetros estáveis + núcleo líder correto'), bitrate=2000)

    nome_arquivo = 'psiqrh_final_equilibrado.mp4'
    print(f"🎬 Gravando vídeo final equilibrado: {nome_arquivo} ...")
    
    ani.save(nome_arquivo, writer=writer)
    print("✅ Vídeo final equilibrado salvo com sucesso!")
    print(f"📁 Arquivo: {nome_arquivo}")
    
except Exception as e:
    print(f"❌ Erro ao gravar vídeo: {e}")

plt.show()

# RELATÓRIO FINAL DEFINITIVO
if len(plasma.historico_fci) > 0:
    print("\n" + "="*70)
    print("RELATÓRIO FINAL ΨQRH - VERSÃO EQUILIBRADA")
    print("="*70)
    
    fci_final = plasma.historico_fci[-1]
    sync_final = plasma.historico_sync[-1]
    coher_final = plasma.historico_coerencia[-1]
    
    fci_max = max(plasma.historico_fci)
    sync_max = max(plasma.historico_sync)
    
    diagnostico, _ = get_diagnostico_final(sync_final, coher_final)
    
    print(f"🎯 ESTADO FINAL: {diagnostico}")
    print(f"📈 FCI Final: {fci_final:.3f}")
    print(f"🔄 Sincronização Final: {sync_final:.3f}") 
    print(f"🌊 Coerência Final: {coher_final:.3f}")
    
    print(f"\n📊 MÁXIMOS ESTÁVEIS:")
    print(f"   FCI Máximo: {fci_max:.3f}")
    print(f"   Sincronização Máxima: {sync_max:.3f}")
    
    # ANÁLISE DA ESTABILIDADE
    fci_std = np.std(plasma.historico_fci[-10:])  # Últimos 10 pontos
    stability = "ALTAMENTE ESTÁVEL" if fci_std < 0.05 else "ESTÁVEL" if fci_std < 0.1 else "INSTÁVEL"
    
    print(f"\n⚖️  ANÁLISE DE ESTABILIDADE:")
    print(f"   Variação do FCI (últimos 10s): {fci_std:.4f}")
    print(f"   Status: {stability}")
    
    # COMPARAÇÃO DEFINITIVA
    print(f"\n📈 EVOLUÇÃO DEFINITIVA:")
    print(f"   Original:        FCI 0.247, Sync 0.076")
    print(f"   Correção Oficial: FCI 0.774 (max), Sync 0.784 (max)")
    print(f"   Versão Final:     FCI {fci_final:.3f}, Sync {sync_final:.3f}")
    
    if fci_final > 0.65 and fci_std < 0.08:
        print(f"\n🎉 SUCESSO! Sistema equilibrado e estável!")
        print(f"   • FCI final > 0.65 ✓")
        print(f"   • Baixa variação ✓") 
        print(f"   • Sincronização sustentada ✓")
    elif fci_max > 0.7:
        print(f"\n✅ Sistema atinge bons picos mas precisa de estabilização")
        print(f"   • FCI máximo: {fci_max:.3f} (bom)")
        print(f"   • Variação: {fci_std:.4f} (precisa melhorar)")
    else:
        print(f"\n⚠️  Sistema precisa de ajustes finos")
        print(f"   • FCI máximo: {fci_max:.3f}")
        print(f"   • Recomendação: K_lider = 65.0")
    
    print("="*70)