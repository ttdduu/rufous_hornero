library(ggplot2)
library(emmeans)
#base_path <- paste0("/home/ttdduu/furnarius-drive-rclone/resultados-figs/",tipo,"/silhouette")
# copio el k=8 del dataset usado para la tesis en datos_crudos_silhouette.csv
# y copio la variable lista_kmeans_score_nulo obtenida cuando ploteo los silhouettes
# {{{ alfa

tipo<-"alfa"
base_path <- paste0("/home/ttdduu/lsd/tesislab/entrenamientos/metricas/sils/",tipo,"/comparacion_nulo_bien_entrenado")

list1<-c(0.81427723,0.7983319,0.8102484,0.7907802,0.8196029,0.80734193,0.80673087,0.79375064,0.82746583,
		 0.7939288,0.78228533,0.803461,0.782626,0.80424464,0.83731645,0.85002434,0.82225025,0.80455333,
		 0.82348496,0.78511363,0.8184547,0.837399,0.77919436,0.78431654,0.8034046,0.8294105,0.83884502,
		 0.80113082,0.80380425)

list2 <- c(0.5205340, 0.5681534, 0.5414522, 0.544548, 0.6001621, 0.5808174, 0.5566903, 0.5373562, 0.5234559,
		   0.4984320, 0.4894437, 0.5165046, 0.5065693)
# }}}

# {{{ machos

tipo<-"macho"
base_path <- paste0("/home/ttdduu/lsd/tesislab/entrenamientos/metricas/sils/",tipo,"/comparacion_nulo_bien_entrenado")

list1<-c(0.6208606,0.6698495,0.6653105,0.6527708,0.6751174,0.64639395,0.6170854,
		 0.67238057,0.6551421,0.61525375,0.6638317,0.6126027,0.6051273,0.62321436,
		 0.65451026,0.63433987,0.6604004,0.6653289,0.6883804,0.6159717,0.6244947,
		 0.6119655,0.5814367,0.62279737,0.6276789,0.64774615,0.62502646,0.66722184,
		 0.69061565)

# copio la variable lista_kmeans_score_nulo obtenida cuando ploteo los silhouettes
list2 <- c(
0.52915376, 0.56861776, 0.5576117, 0.59568703, 0.56092477, 0.6079691, 0.580918,
0.554858, 0.57560885, 0.53244424, 0.52825934, 0.5375298, 0.5380456, 0.61973345
)


# }}}

# {{{ beta

tipo<-"beta"
base_path <- paste0("/home/ttdduu/lsd/tesislab/entrenamientos/metricas/sils/",tipo,"/comparacion_nulo_bien_entrenado")

list1 <- c(0.595833, 0.551205, 0.582826, 0.653979, 0.597157, 0.557390, 0.562801, 0.590180, 0.543861, 0.602345, 0.625790, 0.570483, 0.638518, 0.650956, 0.623955, 0.576548, 0.515074, 0.599873, 0.544133, 0.593552, 0.629862, 0.555897, 0.552721)

list2 <- c(0.5375153, 0.5979301, 0.5475976, 0.49864683, 0.6436776, 0.5888818, 0.6022661, 0.5257627, 0.6028472, 0.5463348, 0.55710685, 0.6346395, 0.5404467, 0.6109579, 0.5778214, 0.63698417, 0.627192, 0.6524898)
# }}}

# {{{ lm

# Combine the lists and create a grouping variable
datos <- data.frame(value = c(list1, list2),
                   group = factor(c(rep("modelo bien entrenado", length(list1)),
                                    rep("modelo nulo", length(list2)))))


shap <- shapiro.test(r1) #pruebo normalidad analíticamente

if (length(list1) > length(list2)) {
  # Calculate the difference in lengths
  difference <- length(list1) - length(list2)
  # Randomly sample indices to remove from list1
  indices_to_remove <- sample(length(list1), difference)
  # Remove the sampled indices from list1
  list1 <- list1[-indices_to_remove]
} else if (length(list2) > length(list1)) {
  # Calculate the difference in lengths
  difference <- length(list2) - length(list1)
  # Randomly sample indices to remove from list2
  indices_to_remove <- sample(length(list2), difference)
  # Remove the sampled indices from list2
  list2 <- list2[-indices_to_remove]
}

model <- lm(value ~ group, data = datos)

r1 <-resid(model, type = "pearson")

boxplot(r1~datos$group,xlab="modelo bien entrenado y modelo nulo", ylab="Residuos estandarizados")
#dev.off()
p<- ggplot(datos, aes(x=group, y=value))+
geom_boxplot()+ labs(y= "Age (years)")+
geom_jitter(aes(color = group))+
theme(legend.position = "none")

summary(model)
shap
emm <- emmeans(model, ~ group)
options(emmeans= list(emmeans = list(infer = c(TRUE, TRUE)),contrast = list(infer = c(TRUE, TRUE))))
# Compute the contrast and its confidence interval
con <- contrast(emm, method = "pairwise", by = NULL,digits=3)
con
# }}}
write.csv(format(as.data.frame(con),digits=3), file = paste0(base_path,"/medias_marginales_nulo_vs_real.csv"))
shap_df <- data.frame(W = shap[["statistic"]], pvalue = shap[["p.value"]])
write.csv(shap_df, file = paste0(base_path, "/shapiro.csv"), row.names = FALSE)
