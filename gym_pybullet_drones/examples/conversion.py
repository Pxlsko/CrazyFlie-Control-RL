from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import math

# Rutas a los archivos de eventos de ambos runs
#event_file_1 = r'C:\Users\pablo\OneDrive\Escritorio\UNI\DRL\MARL\CPP\Single\SingleAgentLogs\ppo_run_1\events.out.tfevents.1746426030.DESKTOP-1E75093.19556.0'
#event_file_5 = r'C:\Users\pablo\OneDrive\Escritorio\UNI\DRL\MARL\CPP\Single\SingleAgentLogs\ppo_run_5\events.out.tfevents.1747047953.HWPC0424.45108.0'
# event_file_1 = r'C:\Users\pablo\OneDrive\Escritorio\UNI\DRL\MARL\CPP\marl_logs\PPO_3\events.out.tfevents.1747218565.HWPC0424.20052.0'

# event_file_1 = r'C:\Users\pablo\OneDrive\Escritorio\UNI\4CARRERA\RL\TFGPython\gym-pybullet-drones\gym_pybullet_drones\examples\results\ComparativaSinErroenObs\tb\PPO_3\events.out.tfevents.1751214136.DESKTOP-1E75093.11336.0'
event_file_1 = r'C:\Users\pablo\OneDrive\Escritorio\UNI\4CARRERA\RL\TFGPython\gym-pybullet-drones\gym_pybullet_drones\examples\results\Trajectoryresultsv8\tb\PPO_5\events.out.tfevents.1751183758.DESKTOP-1E75093.11636.0'

# event_file_5 = r'C:\Users\pablo\OneDrive\Escritorio\UNI\DRL\MARL\CPP\marl_logs\PPO_4\events.out.tfevents.1747304731.HWPC0424.25040.0'

train_tag_names = {
    'train/approx_kl': 'KL aproximada',
    'train/clip_fraction': 'Fracción de actualizaciones',
    'train/clip_range': 'Rango de recorte',
    'train/learning_rate': 'Tasa de aprendizaje',
    'train/entropy_loss': 'Pérdida de entropía',
    'train/explained_variance': 'Varianza explicada',
    'train/loss': 'Pérdida',
    'train/value_loss': 'Pérdida de valor',
    'train/policy_gradient_loss': 'Pérdida de gradiente de política',
    'train/std': 'Desviación típica'
}

# Diccionario de nombres personalizados para los tags de evaluación
eval_tag_names = {
    'eval/mean_ep_length': 'Longitud media del episodio (pasos)',
    'eval/mean_reward': 'Recompensa media'
}

# Carga los datos de ambos runs
event_acc_1 = EventAccumulator(event_file_1)
event_acc_1.Reload()
# event_acc_5 = EventAccumulator(event_file_5)
# event_acc_5.Reload()

# Encuentra los tags comunes
tags_1 = set(event_acc_1.Tags()['scalars'])
# tags_5 = set(event_acc_5.Tags()['scalars'])
# common_tags = sorted(tags_1 & tags_5)
common_tags = sorted(tags_1)

# Filtra los tags de train y eval
train_tags = [tag for tag in common_tags if tag.startswith('train/')]
eval_tags = [tag for tag in common_tags if tag.startswith('eval/')]

print("Tags de entrenamiento:", train_tags)
print("Tags de eval:", eval_tags)

# ----------- Figura para train -----------
n_train = len(train_tags)
cols_train = 2
rows_train = math.ceil(n_train / cols_train)

fig_train, axes_train = plt.subplots(rows_train, cols_train, figsize=(12, 4 * rows_train))
axes_train = axes_train.flatten() if n_train > 1 else [axes_train]

MAX_STEP = 6_500_000  # Máximo de pasos a mostrar

# for i, tag in enumerate(train_tags):
#     events_1 = event_acc_1.Scalars(tag)
#     # events_5 = event_acc_5.Scalars(tag)
#     steps_1 = [e.step for e in events_1]
#     values_1 = [e.value for e in events_1]
#     # steps_5 = [e.step for e in events_5]
#     # values_5 = [e.value for e in events_5]

#     # Filtrar hasta MAX_STEP
#     steps_1_f, values_1_f = zip(*[(s, v) for s, v in zip(steps_1, values_1) if s <= MAX_STEP]) if steps_1 else ([], [])
#     # steps_5_f, values_5_f = zip(*[(s, v) for s, v in zip(steps_5, values_5) if s <= MAX_STEP]) if steps_5 else ([], [])

#     ax = axes_train[i]
#     ax.plot(steps_1_f, values_1_f)
#     # ax.plot(steps_5_f, values_5_f, label='PPO_4')
#     ax.set_ylabel(tag)
#     ax.set_title(tag, fontsize=10)
#     ax.legend(fontsize=8, loc='upper right')
#     ax.grid(True)
fig_train, axes_train = plt.subplots(rows_train, cols_train, figsize=(12, 4 * rows_train))
axes_train = axes_train.flatten() if n_train > 1 else [axes_train]
fig_train.suptitle("Métricas del entrenamiento", fontsize=14)

for i, tag in enumerate(train_tags):
    events_1 = event_acc_1.Scalars(tag)
    steps_1 = [e.step for e in events_1]
    values_1 = [e.value for e in events_1]
    steps_1_f, values_1_f = zip(*[(s, v) for s, v in zip(steps_1, values_1) if s <= MAX_STEP]) if steps_1 else ([], [])

    title_label = train_tag_names.get(tag, tag)  # Título traducido
    y_label = tag  # Mantener el tag original en el eje Y

    ax = axes_train[i]
    ax.plot(steps_1_f, values_1_f)
    ax.set_ylabel(y_label)
    ax.set_title(title_label, fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True)

for j in range(i + 1, len(axes_train)):
    axes_train[j].axis('off')

fig_train.tight_layout(rect=[0, 0, 1, 0.97])
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Aumenta el espacio entre subplots
fig_train.savefig(r'C:\Users\pablo\Downloads\train_tags2.png', bbox_inches='tight')  # Guarda la figura en Descargas
plt.show()

# ----------- Figura para eval -----------
n_eval = len(eval_tags)
cols_eval = 2
rows_eval = math.ceil(n_eval / cols_eval)

fig_eval, axes_eval = plt.subplots(rows_eval, cols_eval, figsize=(12, 4 * rows_eval))
axes_eval = axes_eval.flatten() if n_eval > 1 else [axes_eval]

# for i, tag in enumerate(eval_tags):
#     events_1 = event_acc_1.Scalars(tag)
#     # events_5 = event_acc_5.Scalars(tag)
#     steps_1 = [e.step for e in events_1]
#     values_1 = [e.value for e in events_1]
#     # steps_5 = [e.step for e in events_5]
#     # values_5 = [e.value for e in events_5]

#     # Filtrar hasta MAX_STEP
#     steps_1_f, values_1_f = zip(*[(s, v) for s, v in zip(steps_1, values_1) if s <= MAX_STEP]) if steps_1 else ([], [])
#     # steps_5_f, values_5_f = zip(*[(s, v) for s, v in zip(steps_5, values_5) if s <= MAX_STEP]) if steps_5 else ([], [])

#     # ax = axes_eval[i]
#     # ax.plot(steps_1_f, values_1_f)
#     # # ax.plot(steps_5_f, values_5_f, label='PPO_4')
#     # ax.set_xlabel('Step')
#     # ax.set_ylabel(tag)
#     # ax.set_title(tag)
#     # ax.legend()
#     # ax.grid(True)
#     tag_label = eval_tag_names.get(tag, tag)

#     ax = axes_eval[i]
#     ax.plot(steps_1_f, values_1_f)
#     ax.set_xlabel('Step')
#     ax.set_ylabel(tag_label)
#     ax.set_title(tag_label)
#     ax.legend()
#     ax.grid(True)
fig_eval, axes_eval = plt.subplots(rows_eval, cols_eval, figsize=(12, 4 * rows_eval))
axes_eval = axes_eval.flatten() if n_eval > 1 else [axes_eval]
fig_eval.suptitle("Resultados del entrenamiento", fontsize=14)

for i, tag in enumerate(eval_tags):
    events_1 = event_acc_1.Scalars(tag)
    steps_1 = [e.step for e in events_1]
    values_1 = [e.value for e in events_1]
    steps_1_f, values_1_f = zip(*[(s, v) for s, v in zip(steps_1, values_1) if s <= MAX_STEP]) if steps_1 else ([], [])

    tag_label = eval_tag_names.get(tag, tag)

    ax = axes_eval[i]
    ax.plot(steps_1_f, values_1_f)
    ax.set_xlabel('Step')
    ax.set_ylabel(tag_label)
    ax.set_title(tag_label)
    ax.legend()
    ax.grid(True)


for j in range(i + 1, len(axes_eval)):
    axes_eval[j].axis('off')

fig_eval.tight_layout()
fig_eval.savefig(r'C:\Users\pablo\Downloads\eval_tags2.png', bbox_inches='tight')  # Guarda la figura en Descargas

plt.show()