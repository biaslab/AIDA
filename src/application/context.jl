using AIDA

mutable struct ContextClassifier
    models 
    priors
    vmpits

    function ContextClassifier(models, priors, vmpits=25)
        new(models, priors, vmpits)
    end
end

infer_context(classifier, segment) =  model_selection(segment, classifier.models, classifier.priors, vmp_iter=classifier.vmpits, verbose=true)

# These priors are extracted from silent frames of BABBLE and TRAIN contexts respectively
PRIORS = [Dict(:mθ => [1.0526046070458872, -0.4232782070078879], 
               :vθ => [0.0002274117502010668 -0.0001681986150731712; -0.0001681986150731712 0.00022744882724672668], 
               :aγ => 5144.5, :bγ => 1.5403421819348209, 
               :τ  => 1e12, :order=>2),
          Dict(:mθ => [0.497019359872337, -0.15475030421215585], 
               :vθ => [8.002457029876353e-5 -3.4446944242771045e-5; -3.4446944242771045e-5 8.003099010678723e-5], 
               :aγ => 7679.5, :bγ => 4.857063799328507, 
               :τ  => 1e12, :order=>2)]